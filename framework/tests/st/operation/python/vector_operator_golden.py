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

"""vector op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
"""

import copy
import json
import logging
import math
from pathlib import Path
import random
import struct
import sys
from typing import Any, Dict, List, NamedTuple

import numpy as np
import torch
import torch._prims as prims
import torch.nn.functional as F  # noqa: N812

g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
if str(g_ctrl_path) not in sys.path:
    sys.path.append(str(g_ctrl_path))
from golden_register import GoldenRegister  # noqa: E402

import_path: Path = Path(g_src_root, "framework/tests/cmake/scripts/helper").resolve()
if str(import_path) not in sys.path:
    sys.path.append(str(import_path))
from test_case_desc import TensorDesc  # noqa: E402
from test_case_loader import TestCaseLoader  # noqa: E402
from test_case_tools import (  # noqa: E402
    get_dtype_by_name,
    parse_dict_str,
    parse_list_str,
    str_to_bool,
)

bfloat16 = get_dtype_by_name("bf16", False, False)


def trans_nd_to_fractal_nz(data: np.ndarray, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == np.int8:
        m0, n0 = 16, 32
    elif data.dtype == np.float16 or data.dtype == bfloat16 or data.dtype == np.int32:
        m0, n0 = 16, 16
    else:
        m0, n0 = 16, 8
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    else:
        data = np.pad(data, (batch_padding + ((0, 0), (0, padding_n))), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).transpose(*array_trans)
    return data


def dump_file(data_pool, data_path, type_str):
    np.array(data_pool).astype(get_dtype_by_name(type_str.lower())).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)


def load_test_cases_from_json(json_file: str) -> list:
    with open(json_file, "r") as data_file:
        json_data = json.load(data_file)
    if json_data is None:
        raise ValueError(f"Json file {json_file} is invalid.")
    if "test_cases" in json_data:
        test_cases = json_data["test_cases"]
    else:
        test_cases = [json_data]
    test_cases.sort(key=lambda x: x["case_index"])
    return test_cases


def _generate_golden_input_tensor(op: str, input_tensor: dict, config: dict, index: int, spec_value_map: dict):
    min_value = input_tensor["data_range"]["min"]
    max_value = input_tensor["data_range"]["max"]
    dtype = get_dtype_by_name(input_tensor["dtype"])
    if op == "QuantMX":
        return _generate_quantmx_input(input_tensor, config)
    if op == "Cast" and input_tensor["dtype"] in ("fp4_e2m1", "fp4_e1m2"):
        values = gen_uniform_data(input_tensor["shape"], min_value, max_value, np.float32)
        encode = _encode_e2m1_vectorized if input_tensor["dtype"] == "fp4_e2m1" else _encode_e1m2_cast
        return _pack_fp4_e2m1x2_low_first(encode(values))
    if min_value != max_value:
        assert not isinstance(min_value, str) and not isinstance(max_value, str), (
            "Data range must be number when the min and max are not same."
        )
        if op == "ScatterUpdate" and index == 1:
            return np.random.choice(range(min_value, max_value), input_tensor["shape"], False).astype(dtype)
        return np.random.uniform(min_value, max_value, input_tensor["shape"]).astype(dtype)
    if isinstance(min_value, str):
        assert min_value in spec_value_map.keys(), f"Data range of input tensor {input_tensor} has invalid value."
        max_value = spec_value_map.get(max_value)
    return np.full(input_tensor["shape"], max_value, dtype=dtype)


def _build_golden_input_tensors(op: str, config: dict) -> list:
    spec_value_map = {
        "nan": np.nan,
        "inf": np.inf,
        "-inf": -np.inf,
        "max": np.finfo(np.float32).max,
        "min": np.finfo(np.float32).min,
    }
    return [
        _generate_golden_input_tensor(op, input_tensor, config, index, spec_value_map)
        for index, input_tensor in enumerate(config["input_tensors"])
    ]


def _write_golden_inputs(input_tensors: list, output_path: Path, config: dict) -> None:
    cube_op_list = ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]
    for input_tensor, read_input in zip(input_tensors, config["input_tensors"]):
        if config.get("operation") in cube_op_list and read_input.get("format") == "NZ":
            input_tensor = trans_nd_to_fractal_nz(input_tensor)
        input_tensor.tofile(Path(output_path, read_input["name"] + ".bin"))


def _write_golden_outputs(res: list, output_path: Path, config: dict) -> None:
    for idx in range(len(config["output_tensors"])):
        output_dtype = config["output_tensors"][idx]["dtype"]
        output_file = Path(output_path, config["output_tensors"][idx]["name"] + ".bin")
        if output_dtype in ["fp8e4m3", "fp8e5m2", "fp8e8m0", "hf8", "fp4_e2m1x2"] and res[idx].dtype == np.uint8:
            res[idx].tofile(output_file)
        else:
            res[idx].astype(get_dtype_by_name(output_dtype)).tofile(output_file)


def get_c_sizeof_type(type_str):
    type_dict = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "int32": 4,
        "int16": 2,
        "int8": 1,
    }
    return type_dict[type_str]


def gen_op_golden(op: str, golden_func, output_path: Path, case_index: int = None) -> bool:
    def generate_golden_files(golden_func, output_path: Path, config: dict) -> bool:
        if config['operation'] in ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]:
            return generate_matmul_golden_files(golden_func, output_path, config)

        input_tensors = _build_golden_input_tensors(op, config)
        res = golden_func(input_tensors, config)
        _write_golden_inputs(input_tensors, output_path, config)
        _write_golden_outputs(res, output_path, config)
        return True

    case_path: Path = Path(Path(__file__).parent.parent, "test_case").resolve()
    case_file: Path = Path(case_path, op + "_st_test_cases.json").resolve()
    test_configs = load_test_cases_from_json(str(case_file))
    if len(test_configs) == 0:
        raise ValueError("Not find test cases, please check.")

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, test_config in enumerate(test_configs):
            output_path1 = Path(output_path, str(index))
            output_path1.mkdir(parents=True, exist_ok=True)
            generate_golden_files(golden_func, output_path1, test_config)
    else:
        generate_golden_files(golden_func, output_path, test_configs[case_index])
    return True


def generate_matmul_params_files(input_tensors: list, output_path: Path, config: dict):
    params = config.get("params")
    if params.get("scale_tensors"):
        scale_tensor = params["scale_tensors"]
        scale_min = scale_tensor["data_range"]["min"]
        scale_max = scale_tensor["data_range"]["max"]
        if scale_min != scale_max:
            tensor = np.random.uniform(scale_min, scale_max, scale_tensor["shape"]).astype(np.float32)
        else:
            tensor = np.full(
                scale_tensor["shape"],
                scale_max,
                dtype=np.float32,
            )
        tensor_data = tensor.view(np.uint32)
        mask = 0xFFFFE000
        tensor_data = tensor_data & mask
        fp32_modified = tensor_data.view(np.float32)
        tensor_data.astype(np.uint64).tofile(Path(output_path, params["scale_tensors"]["name"] + ".bin"))
        input_tensors.append(fp32_modified)

    if params.get("bias_tensors"):
        bias_tensor = params["bias_tensors"]
        bias_min = bias_tensor["data_range"]["min"]
        bias_max = bias_tensor["data_range"]["max"]
        tensor_type = get_dtype_by_name(bias_tensor["dtype"])
        if bias_tensor["dtype"] == "bf16":
            tensor_type = bfloat16
        tensor = np.random.uniform(bias_min, bias_max, bias_tensor["shape"]).astype(tensor_type)
        tensor.tofile(Path(output_path, params["bias_tensors"]["name"] + ".bin"))
        input_tensors.append(tensor)


def generate_matmul_golden_files(golden_func, output_path: Path, config: dict):
    cube_op_list = ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]
    index = 0
    input_tensors = []
    for input_tensor in config["input_tensors"]:
        input_min = input_tensor["data_range"]["min"]
        input_max = input_tensor["data_range"]["max"]
        assert not isinstance(input_min, str) and not isinstance(input_min, str), (
            "Data range must be number when the min and max are not same."
        )
        tensor_type = get_dtype_by_name(input_tensor["dtype"])
        if input_tensor["dtype"] == "bf16":
            tensor_type = bfloat16
        tensor = np.random.uniform(input_min, input_max, input_tensor["shape"]).astype(tensor_type)
        index += 1
        input_tensors.append(tensor)

    generate_matmul_params_files(input_tensors, output_path, config)

    res = golden_func(input_tensors, config)
    for idx in range(len(config["output_tensors"])):
        tensor_type = get_dtype_by_name(config["output_tensors"][idx]["dtype"])
        if config["output_tensors"][idx]["dtype"] == "bf16":
            tensor_type = bfloat16
        res[idx].astype(tensor_type).tofile(Path(output_path, config["output_tensors"][idx]["name"] + ".bin"))

    for input_tensor, read_input in zip(input_tensors, config["input_tensors"]):
        if config.get("operation") in cube_op_list and read_input.get("format") == "NZ":
            input_tensor = trans_nd_to_fractal_nz(input_tensor)
        input_tensor.tofile(Path(output_path, read_input["name"] + ".bin"))

    params = config.get("params")
    l0c2l1_tensor = params.get("l0c2l1_tensor")
    if l0c2l1_tensor:
        # 2：l0c2l1场景res = [tensor_out, tensor_l0c2l1_data]
        assert len(res) == 2
        res[1].tofile(Path(output_path, l0c2l1_tensor["name"] + ".bin"))
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatterUpdate/ScatterUpdateOperationTest.TestScatterUpdate",
    ]
)
def gen_scatter_update_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, config):
        src = inputs[0]
        index = inputs[1]
        dst = inputs[2]
        axis = src.ndim

        # 自测，src = np.random.randint(2, 3, src.shape).astype(np.float32)
        # 自测，dst = np.random.randint(1, 2, dst.shape).astype(np.float32)
        if axis == 4:
            b = index.shape[0]
            s = index.shape[1]
            block_size = dst.shape[1]
            result = copy.copy(dst)
            for _b in range(b):
                for _s in range(s):
                    result[index[_b][_s] // block_size][index[_b][_s] % block_size][:] = src[_b][_s][:]
        elif axis == 2:
            b = index.shape[0]
            s = index.shape[1]
            result = copy.copy(dst)

            for _b in range(b):
                for _s in range(s):
                    result[index[_b][_s]][:] = src[_b * s + _s][:]
        else:
            logging.error("axis ERROR!")

        logging.debug("src:")
        logging.debug(inputs[0])
        logging.debug("index:")
        logging.debug(inputs[1])
        logging.debug("dst:")
        logging.debug(inputs[2])
        logging.debug("result:")
        logging.debug(result)

        return [result]

    logging.info("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ScatterUpdate", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"])
def matmul_params_func(params: dict):
    bias_params_func(params)
    fixpipe_params_func(params)
    l0c2l1_params_func(params)
    return params


def fixpipe_params_func(params: dict):
    scale_shape = [1, 1]
    scale_range = [1, 1]
    scale_dtype = "uint64"
    fixpipe_param = params.get("fixpipe_info", "")
    if fixpipe_param is None:
        params["fixpipe_info"] = ""
    if fixpipe_param != "" and fixpipe_param is not None:
        fixpipe_info = parse_dict_str(fixpipe_param)
        if "scale_value" in fixpipe_info:
            params["scale_value"] = float(fixpipe_info["scale_value"])
        if "relu_type" in fixpipe_info:
            params["relu_type"] = int(fixpipe_info["relu_type"])
        if "scale_tensor_range" in fixpipe_info and fixpipe_info["scale_tensor_range"] != "":
            scale_range = parse_list_str(fixpipe_info["scale_tensor_range"])
        if "quant_type" in fixpipe_info:
            params["quant_type"] = int(fixpipe_info["quant_type"])
        if "scale_shape" in fixpipe_info:
            scale_shape_info = parse_list_str(fixpipe_info["scale_shape"])
            if scale_shape_info[0] == 0 or len(scale_shape_info) < 2:
                scale_range = [1, 1]
            else:
                scale_shape = scale_shape_info
    # scale_tensor
    scale_tensor = TensorDesc(
        "scale_tensor", scale_shape, scale_dtype, scale_range, tensor_format="ND", need_trans=False
    )
    params["scale_tensors"] = scale_tensor.dump_to_json()


def bias_params_func(params: dict):
    bias_shape = [1, 1]
    bias_range = [0, 0]
    bias_dtype = "fp16"
    bias_param = params.get("bias_info", "")
    if bias_param is None:
        params["bias_info"] = ""
    if bias_param != "" and bias_param is not None:
        bias_info = parse_dict_str(bias_param)
        if "bias_range" in bias_info:
            bias_range = parse_list_str(bias_info["bias_range"])
        if "bias_dtype" in bias_info:
            bias_dtype = bias_info["bias_dtype"]
        if "bias_shape" in bias_info:
            bias_shape_info = parse_list_str(bias_info["bias_shape"])
            if bias_shape_info[0] == 0 or len(bias_shape_info) < 2:
                bias_range = [0, 0]
            else:
                bias_shape = bias_shape_info
    # bias_tensor
    bias_tensor = TensorDesc("bias_tensor", bias_shape, bias_dtype, bias_range, tensor_format="ND", need_trans=False)
    params["bias_tensors"] = bias_tensor.dump_to_json()


def l0c2l1_params_func(params: dict):
    l0c2l1_info_params = params.get("l0c2l1_info", None)
    if l0c2l1_info_params is None:
        return
    l0c2l1_info = parse_dict_str(l0c2l1_info_params)
    required_keys = {"input_shape", "input_dtype", "input_format", "is_trans", "is_as_left_matrix"}
    if not required_keys.issubset(l0c2l1_info):
        raise ValueError("l0c2l1_info params is invalid, please check!")
    l0c2l1_is_as_left_matrix = str_to_bool(l0c2l1_info["is_as_left_matrix"])
    l0c2l1_is_trans = str_to_bool(l0c2l1_info.get("is_trans", False))
    l0c2l1_range = parse_list_str(l0c2l1_info.get("input_range", "[-1, 1]"))
    l0c2l1_tensor = TensorDesc(
        "l0c2l1_tensor",
        parse_list_str(l0c2l1_info["input_shape"]),
        l0c2l1_info["input_dtype"],
        l0c2l1_range,
        l0c2l1_info["input_format"],
        need_trans=l0c2l1_is_trans,
    )
    params["l0c2l1_tensor"] = l0c2l1_tensor.dump_to_json()
    params["l0c2l1_params"] = {"is_as_left_matrix": l0c2l1_is_as_left_matrix, "is_l0c2l1_trans": l0c2l1_is_trans}


@TestCaseLoader.reg_params_handler(ops=["Cast"])
def cast_params_func(params: dict):
    params["mode"] = int(params.get("mode", "0"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestOneHot/OneHotOperationTest.TestOneHot",
    ]
)
def gen_onehot_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        num_classes = config["params"]["num_classes"]
        return [np.eye(num_classes, dtype=np.int32)[np.array(inputs[0])]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("OneHot", golden_func, output, case_index)


_FP4_E2M1_DECODE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _pack_fp4_e1m2x2_low_first(codes: np.ndarray) -> np.ndarray:
    return _pack_fp4_e2m1x2_low_first(codes)


def _encode_e1m2_cast(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    abs_val = np.abs(values)
    sign = np.where(values < 0, np.uint8(8), np.uint8(0))
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)

    clipped = np.clip(abs_val, 0.0, 1.75)
    code = np.round(clipped * np.float32(4.0)).astype(np.uint8)
    code = np.clip(code, np.uint8(0), np.uint8(7))

    result = code | sign
    result[nan_mask | inf_mask] = np.uint8(7) | sign[nan_mask | inf_mask]
    return result


def _decode_fp4_e2m1x2(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    shape = list(packed.shape)
    shape[-1] *= 2
    unpacked = np.zeros(shape, dtype=np.float32)
    lo = packed & np.uint8(0x0F)
    hi = packed >> np.uint8(4)
    lo_sign = np.where((lo & np.uint8(8)) != 0, -1.0, 1.0).astype(np.float32)
    hi_sign = np.where((hi & np.uint8(8)) != 0, -1.0, 1.0).astype(np.float32)
    unpacked[..., 0::2] = lo_sign * _FP4_E2M1_DECODE[lo & np.uint8(0x07)]
    unpacked[..., 1::2] = hi_sign * _FP4_E2M1_DECODE[hi & np.uint8(0x07)]
    return unpacked


def _decode_fp4_e1m2x2(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    shape = list(packed.shape)
    shape[-1] *= 2
    mag = np.zeros(shape, dtype=np.uint8)
    mag[..., 0::2] = packed & np.uint8(0x0F)
    mag[..., 1::2] = (packed >> np.uint8(4)) & np.uint8(0x0F)
    sign = np.where((mag & np.uint8(8)) != 0, -1.0, 1.0).astype(np.float32)
    code = (mag & np.uint8(7)).astype(np.int32)
    mant = (code & np.int32(3)).astype(np.float32)
    exp = (code >> np.int32(2)).astype(np.float32)
    normal = exp > 0
    val = np.where(
        normal, (1.0 + mant * np.float32(0.25)) * (np.float32(2.0) ** (exp - np.float32(1.0))), mant * np.float32(0.25)
    )
    return sign * val


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCast/CastOperationTest.TestCast",
    ]
)
def gen_cast_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def _cast_fp4_e2m1x2_encode(inputs_bf16: np.ndarray) -> np.ndarray:
        src = inputs_bf16.astype(np.float32)
        codes = _encode_e2m1_vectorized(src)
        return _pack_fp4_e2m1x2_low_first(codes)

    def _cast_fp4_e1m2x2_encode(inputs_bf16: np.ndarray) -> np.ndarray:
        src = inputs_bf16.astype(np.float32)
        codes = _encode_e1m2_cast(src)
        return _pack_fp4_e1m2x2_low_first(codes)

    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        output_dtype = config.get("output_tensors")[0].get("dtype")
        dst_dtype = params.get("dst_dtype", output_dtype)
        src_dtype = config.get("input_tensors")[0].get("dtype")

        if dst_dtype == "fp4_e2m1":
            return [_cast_fp4_e2m1x2_encode(inputs[0])]
        if dst_dtype == "fp4_e1m2":
            return [_cast_fp4_e1m2x2_encode(inputs[0])]
        if src_dtype == "fp4_e2m1":
            x = _decode_fp4_e2m1x2(inputs[0])
            return [x.astype(get_dtype_by_name(dst_dtype))]
        if src_dtype == "fp4_e1m2":
            x = _decode_fp4_e1m2x2(inputs[0])
            return [x.astype(get_dtype_by_name(dst_dtype))]

        if inputs[0].dtype == bfloat16:
            dtype_out = get_dtype_by_name(dst_dtype)
            x = inputs[0].astype(dtype_out)
        else:
            dtype_out = get_dtype_by_name(dst_dtype, True)
            if dtype_out is None:
                return [inputs[0].astype(get_dtype_by_name(dst_dtype))]
            x = torch.from_numpy(inputs[0])
            if dtype_out == torch.bfloat16:
                x = x.to(torch.float32).numpy().astype(bfloat16)
            elif dst_dtype == "hf8":
                x = inputs[0].astype("hifloat8")
            elif dst_dtype == "fp8e4m3":
                x = inputs[0].astype("float8_e4m3fn")
            elif dst_dtype == "fp8e5m2":
                x = inputs[0].astype("float8_e5m2")
            elif dst_dtype == "fp8e8m0":
                x = inputs[0].astype("float8_e8m0")
            else:
                x = x.to(dtype_out).numpy()

        return [x]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Cast", golden_func, output, case_index)


def l0c2l_golden_generate(inputs: list, config: dict):
    params = config.get("params")
    l0c2l1_tensor = params["l0c2l1_tensor"]
    l0c2l1_min = l0c2l1_tensor["data_range"]["min"]
    l0c2l1_max = l0c2l1_tensor["data_range"]["max"]
    tensor_l0c2l1 = np.random.uniform(l0c2l1_min, l0c2l1_max, l0c2l1_tensor["shape"]).astype(
        get_dtype_by_name(l0c2l1_tensor["dtype"])
    )
    tensor_l0c2l1_data = tensor_l0c2l1
    tensor_a = inputs[0] if not params.get("transA") else np.swapaxes(inputs[0], inputs[0].ndim - 2, inputs[0].ndim - 1)
    tensor_b = inputs[1] if not params.get("transB") else np.swapaxes(inputs[1], inputs[1].ndim - 2, inputs[1].ndim - 1)
    tensor_l0c2l1 = (
        tensor_l0c2l1
        if not l0c2l1_tensor["need_trans"]
        else np.swapaxes(tensor_l0c2l1, tensor_l0c2l1.ndim - 2, tensor_l0c2l1.ndim - 1)
    )

    tensor_tmp = torch.matmul(
        torch.from_numpy(tensor_a.astype(np.float32)).to(torch.float32),
        torch.from_numpy(tensor_b.astype(np.float32)).to(torch.float32),
    ).to(torch.float32)
    if params.get("relu_type") == 1:
        tensor_tmp = F.relu(tensor_tmp)
    if params.get("scale_value"):
        mask = 0xFFFFE000
        scale_data = np.float32([params.get("scale_value")]).view(np.uint32)
        uint32_num = scale_data & mask
        fp32_scale_modified = uint32_num.view(np.float32)[0]
        tensor_tmp = tensor_tmp * fp32_scale_modified
    if params.get("quant_type") is not None and params.get("quant_type") == 2:
        # quant type中no quant为0, pertensor为1, perchannel为2.
        tensor_tmp = tensor_tmp * inputs[2]
    tensor_tmp = tensor_tmp.numpy().astype(get_dtype_by_name(l0c2l1_tensor["dtype"]))
    l0c2l1_is_left_matrix = str_to_bool(params["l0c2l1_params"]["is_as_left_matrix"])
    if l0c2l1_is_left_matrix:
        tensor_out = (
            torch.matmul(
                torch.from_numpy(tensor_l0c2l1.astype(np.float32)).to(torch.float32),
                torch.from_numpy(tensor_tmp.astype(np.float32)).to(torch.float32),
            )
            .to(torch.float32)
            .numpy()
            .astype(get_dtype_by_name(params["outDtype"]))
        )
    else:
        tensor_out = (
            torch.matmul(
                torch.from_numpy(tensor_tmp.astype(np.float32)).to(torch.float32),
                torch.from_numpy(tensor_l0c2l1.astype(np.float32)).to(torch.float32),
            )
            .to(torch.float32)
            .numpy()
            .astype(get_dtype_by_name(params["outDtype"]))
        )

    if l0c2l1_tensor["format"] == "NZ":
        tensor_l0c2l1_data = trans_nd_to_fractal_nz(tensor_l0c2l1_data)

    if params.get("isCMatrixNz"):
        tensor_out = trans_nd_to_fractal_nz(tensor_out, True)

    return [tensor_out, tensor_l0c2l1_data]


def matmul_golden_func(inputs: list, config: dict):
    params = config.get("params")
    if params.get("l0c2l1_tensor"):
        return l0c2l_golden_generate(inputs, config)
    tensor_a = inputs[0] if not params.get("transA") else np.swapaxes(inputs[0], inputs[0].ndim - 2, inputs[0].ndim - 1)
    tensor_b = inputs[1] if not params.get("transB") else np.swapaxes(inputs[1], inputs[1].ndim - 2, inputs[1].ndim - 1)
    assert params.get("outDtype") in ("fp32", "fp16", "bf16", "int32")
    if params.get("outDtype") in ("fp32", "fp16", "bf16"):
        tensor_c = torch.matmul(
            torch.from_numpy(tensor_a.astype(np.float32)).to(torch.float32),
            torch.from_numpy(tensor_b.astype(np.float32)).to(torch.float32),
        ).to(torch.float32)
        if params.get("bias_info") is not None and params.get("bias_info") != "":
            tensor_c = tensor_c + torch.from_numpy(inputs[3].astype(np.float32)).to(torch.float32)
    else:
        tensor_c = torch.matmul(
            torch.from_numpy(tensor_a.astype(np.int32)).to(torch.int32),
            torch.from_numpy(tensor_b.astype(np.int32)).to(torch.int32),
        ).to(torch.int32)
        if params.get("bias_info") is not None and params.get("bias_info") != "":
            tensor_c = tensor_c + torch.from_numpy(inputs[3].astype(np.int32)).to(torch.int32)

    if params.get("relu_type") == 1:
        tensor_c = F.relu(tensor_c)
    if params.get("scale_value"):
        mask = 0xFFFFE000
        scale_value_data = np.float32([params.get("scale_value")]).view(np.uint32) & mask
        fp32_scale_modified = scale_value_data.view(np.float32)[0]
        tensor_c = tensor_c * fp32_scale_modified
    if params.get("quant_type") is not None and params.get("quant_type") == 2:
        # quant type中no quant为0, pertensor为1, perchannel为2.
        tensor_c = tensor_c * inputs[2]
    tensor_c = tensor_c.numpy()
    tensor_type = get_dtype_by_name(params.get("outDtype"))
    if params.get("outDtype") == "bf16":
        tensor_type = bfloat16
    tensor_c = tensor_c.astype(tensor_type)

    if params.get("isCMatrixNz"):
        tensor_c = trans_nd_to_fractal_nz(tensor_c, True)

    return [tensor_c]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExpand/ExpandOperationTest.TestExpand",
    ]
)
def gen_expand_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        output_dtype = config.get("output_tensors")[0].get("dtype")
        output_shape = config.get("output_tensors")[0].get("shape")
        dst_dtype = params.get("dst_dtype", output_dtype)
        dst_shape = params.get("dst_shape", output_shape)
        if inputs[0].dtype == bfloat16:
            dtype_out = get_dtype_by_name(dst_dtype)
            x = inputs[0].astype(dtype_out)
        else:
            dtype_out = get_dtype_by_name(dst_dtype, True)
            if dtype_out is None:
                x = inputs[0].astype(get_dtype_by_name(dst_dtype))
            else:
                x = torch.from_numpy(inputs[0])
                if dtype_out == torch.bfloat16:
                    x = x.to(torch.float32).numpy().astype(bfloat16)
                else:
                    x = x.to(dtype_out).numpy()

        x = np.broadcast_to(x, np.array(dst_shape))
        x = np.copy(x)
        return [x]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Expand", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMatmul/MatmulOperationTest.TestMatmul",
    ],
    version=0,
    timeout=0,
)
def gen_matmul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Matmul", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMatmulVerify/MatmulVerifyOperationTest.TestMatmulVerify",
    ]
)
def gen_matmulverify_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("MatmulVerify", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBatchMatmul/BatchMatmulOperationTest.TestBatchMatmul",
    ],
    version=0,
    timeout=0,
)
def gen_batchmatmul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BatchMatmul", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBatchMatmulVerify/BatchMatmulVerifyOperationTest.TestBatchMatmulVerify",
    ]
)
def gen_batchmatmulverify_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BatchMatmulVerify", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExp/ExpOperationTest.TestExp",
    ]
)
def gen_exp_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.exp(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Exp", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExpm1/Expm1OperationTest.TestExpm1",
    ]
)
def gen_expm1_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        input_dtype = inputs[0].dtype
        y = torch.expm1(x)
        if input_dtype == bfloat16:
            y = y.to(torch.float32).numpy().astype(bfloat16)
        return [np.array(y)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Expm1", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExp2/Exp2OperationTest.TestExp2",
    ]
)
def gen_exp2_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        a = from_numpy(inputs[0])
        c = torch.exp2(a)
        return [to_numpy(c)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Exp2", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAbs/AbsOperationTest.TestAbs",
    ]
)
def gen_abs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        return [np.abs(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Abs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestNeg/NegOperationTest.TestNeg",
    ]
)
def gen_neg_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.negative(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Neg", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLog/LogOperationTest.TestLog",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        base = _config["params"]["base"]
        input_dtype = inputs[0].dtype
        if input_dtype == np.float16:
            inputs[0].astype(np.float32)
        if base == "e":
            output = [np.log(inputs[0])]
        elif base == "2":
            output = [np.log2(inputs[0])]
        elif base == "10":
            output = [np.log10(inputs[0])]
        if input_dtype == np.float16:
            output = [output[0].astype(np.float16)]
        return output

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Log", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLog1p/Log1pOperationTest.TestLog1p",
    ]
)
def gen_log1p_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        input_dtype = inputs[0].dtype
        if input_dtype == np.float16:
            inputs[0].astype(np.float32)

        output = [np.log1p(inputs[0])]
        if input_dtype == np.float16:
            output = [output[0].astype(np.float16)]
        return output

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Log1p", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTan/TanOperationTest.TestTan",
    ]
)
def gen_tan_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs, _config: dict):
        output = [np.tan(inputs[0])]
        return output

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Tan", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPow/PowOperationTest.TestPow",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        a = from_numpy(inputs[0])
        b = from_numpy(inputs[1])
        c = torch.pow(a, b)
        return [to_numpy(c)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Pow", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPack/PackOperationTest.TestPack",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        return [inputs[0].flatten().view(np.uint8)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Pack", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestUnPack/UnPackOperationTest.TestUnPack",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        params = _config.get("params")
        output_dtype = _config.get("output_tensors")[0].get("dtype")
        dst_dtype = params.get("dst_dtype", output_dtype)
        # 映射缩写到 numpy 兼容的 dtype 名
        # 映射缩写: fp16→numpy float16, bf16→项目自定义bfloat16类型
        dtype_map = {"fp16": "float16", "bf16": bfloat16}
        view_dtype = dtype_map.get(dst_dtype, dst_dtype)
        return [inputs[0].view(view_dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("UnPack", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPows/PowsOperationTest.TestPows",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        params = _config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [np.power(inputs[0], params["scalar"])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Pows", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAtan/AtanOperationTest.TestAtan",
    ]
)
def gen_atan_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.arctan(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Atan", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAtan2/Atan2OperationTest.TestAtan2",
    ]
)
def gen_atan2_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.arctan2(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Atan2", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestErf/ErfOperationTest.TestErf",
    ]
)
def gen_erf_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.erf(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Erf", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSin/SinOperationTest.TestSin",
    ]
)
def gen_sin_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        a = from_numpy(inputs[0])
        c = torch.sin(a)
        return [to_numpy(c)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sin", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCos/CosOperationTest.TestCos",
    ]
)
def gen_cos_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        a = from_numpy(inputs[0])
        c = torch.cos(a)
        return [to_numpy(c)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Cos", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestErfc/ErfcOperationTest.TestErfc",
    ]
)
def gen_erfc_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.erfc(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Erfc", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRound/RoundOperationTest.TestRound",
    ]
)
def gen_round_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        decimals = 0 if not params.get("decimals") else params["decimals"]
        input_dtype = inputs[0].dtype
        tensor_dtype = get_dtype_by_name(input_dtype, True)
        if np.issubdtype(input_dtype, np.integer):
            y = np.around(inputs[0], decimals=decimals)
            return [np.array(y)]
        x = safe_tensor_conversion(inputs[0])
        x = x.to(torch.float32)
        y = torch.round(x, decimals=decimals)
        if input_dtype == bfloat16:
            y = y.to(torch.float32).numpy().astype(bfloat16)
        else:
            y = y.to(tensor_dtype).numpy()
        return [np.array(y)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Round", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRsqrt/RsqrtOperationTest.TestRsqrt",
    ]
)
def gen_rsqrt_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [1 / np.sqrt(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Rsqrt", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSign/SignOperationTest.TestSign",
    ]
)
def gen_sign_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        x = torch.sign(x)
        return [to_numpy(x)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sign", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSignbit/SignbitOperationTest.TestSignbit",
    ]
)
def gen_signbit_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        x = torch.signbit(x)
        return [to_numpy(x)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Signbit", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTanh/TanhOperationTest.TestTanh",
    ]
)
def gen_tanh_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        x = torch.tanh(x)
        return [to_numpy(x)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Tanh", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTan/TanOperationTest.TestTan",
    ]
)
def gen_tan_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        x = torch.tan(x)
        return [to_numpy(x)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Tan", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRelu/ReluOperationTest.TestRelu",
    ]
)
def gen_relu_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.maximum(inputs[0], 0)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Relu", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCeil/CeilOperationTest.TestCeil",
    ]
)
def gen_ceil_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.ceil(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Ceil", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFloor/FloorOperationTest.TestFloor",
    ]
)
def gen_floor_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.floor(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Floor", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTrunc/TruncOperationTest.TestTrunc",
    ]
)
def gen_trunc_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.trunc(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Trunc", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPad/PadOperationTest.TestPad",
    ]
)
def gen_pad_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        input_shape = config["input_tensors"][0]["shape"]
        output_shape = config["output_tensors"][0]["shape"]
        pad_value_type = config.get("params", {}).get("pad_value_type", "zero")
        if pad_value_type == "min":
            pad_value = -torch.inf
        elif pad_value_type == "max":
            pad_value = torch.inf
        elif pad_value_type == "custom":
            pad_value = config.get("params", {}).get("pad_value", 0.0)
        else:
            pad_value = 0.0
        if inputs[0].dtype == bfloat16:
            tensor = torch.from_numpy(inputs[0].astype(np.float32)).to(torch.bfloat16)
        else:
            tensor = torch.from_numpy(inputs[0])

        if len(input_shape) == 1:
            pad_right = output_shape[-1] - input_shape[-1]
            result = F.pad(tensor, (0, pad_right), mode='constant', value=pad_value)
        else:
            pad_right = output_shape[-1] - input_shape[-1]
            pad_bottom = output_shape[-2] - input_shape[-2]
            result = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
        return [to_numpy(result)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Pad", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFillPad/FillPadOperationTest.TestFillPad",
    ]
)
def gen_fillpad_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        input_shape = config["input_tensors"][0]["shape"]
        valid_shape = config["view_shape"]
        output_shape = config["output_tensors"][0]["shape"]
        pad_value_type = config.get("params", {}).get("pad_value_type", "zero")
        if pad_value_type == "min":
            pad_value = -np.inf
        elif pad_value_type == "max":
            pad_value = np.inf
        elif pad_value_type == "custom":
            pad_value = config.get("params", {}).get("pad_value", 0.0)
        else:
            pad_value = 0.0

        input_tensor = inputs[0]
        result = np.full(output_shape, pad_value, dtype=input_tensor.dtype)

        ndim = len(input_shape)
        if ndim == 1:
            result[:valid_shape[0]] = input_tensor[:valid_shape[0]]
        elif ndim == 2:
            result[:valid_shape[0], :valid_shape[1]] = input_tensor[:valid_shape[0], :valid_shape[1]]
        elif ndim == 3:
            result[:valid_shape[0], :valid_shape[1], :valid_shape[2]] = input_tensor[
                :valid_shape[0], :valid_shape[1], :valid_shape[2]
            ]
        elif ndim == 4:
            result[:valid_shape[0], :valid_shape[1], :valid_shape[2], :valid_shape[3]] = input_tensor[
                :valid_shape[0], :valid_shape[1], :valid_shape[2], :valid_shape[3]
            ]

        return [result]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("FillPad", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSqrt/SqrtOperationTest.TestSqrt",
    ]
)
def gen_sqrt_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.sqrt(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sqrt", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestReciprocal/ReciprocalOperationTest.TestReciprocal",
    ]
)
def gen_reciprocal_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.reciprocal(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Reciprocal", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseNot/BitwiseNotOperationTest.TestBitwiseNot",
    ]
)
def gen_bitwise_not_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        assert len(inputs) > 0, "inputs must contain at least one element"

        x = from_numpy(inputs[0]).npu()
        result = torch.bitwise_not(x)
        result = result.cpu()
        result_np = to_numpy(result)
        return [result_np]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseNot", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdd/AddOperationTest.TestAdd",
    ]
)
def gen_add_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] + inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Add", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestHypot/HypotOperationTest.TestHypot",
    ]
)
def gen_hypot_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.hypot(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Hypot", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPReLU/PReLUOperationTest.TestPReLU",
    ]
)
def gen_prelu_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, _config: dict):
        x = inputs[0]
        weight = inputs[1]
        is_bfloat16 = x.dtype == bfloat16

        if is_bfloat16:
            x_tensor = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)
            weight_tensor = torch.from_numpy(weight.astype(np.float32)).to(torch.bfloat16)
        else:
            x_tensor = torch.from_numpy(x)
            weight_tensor = torch.from_numpy(weight)

        result_tensor = F.prelu(x_tensor, weight_tensor)

        return [to_numpy(result_tensor)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("PReLU", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFmod/FmodOperationTest.TestFmod",
    ]
)
def gen_fmod_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        y = torch.fmod(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Fmod", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFmods/FmodsOperationTest.TestFmods",
    ]
)
def gen_fmods_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(params["scalar"])
        y = torch.fmod(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Fmods", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLogicalNot/LogicalNotOperationTest.TestLogicalNot",
    ]
)
def gen_logical_not_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        if x.dtype in (torch.uint16, torch.uint32):
            y = np.logical_not(x.numpy())
        else:
            y = torch.logical_not(x)
        return [np.array(y)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("LogicalNot", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLogicalAnd/LogicalAndOperationTest.TestLogicalAnd",
    ]
)
def gen_logical_and_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = safe_tensor_conversion(inputs[0])
        x1 = safe_tensor_conversion(inputs[1])
        y = torch.logical_and(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("LogicalAnd", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSub/SubOperationTest.TestSub",
    ]
)
def gen_sub_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] - inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sub", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseAnd/BitwiseAndOperationTest.TestBitwiseAnd",
    ]
)
def gen_bitwise_and_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        assert len(inputs) > 1, "inputs must contain at least two elements"
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        if x0.dtype == torch.uint16:
            x0.numpy()
            x1.numpy()
            y = np.bitwise_and(x0, x1)
        else:
            y = torch.bitwise_and(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseAnd", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseOr/BitwiseOrOperationTest.TestBitwiseOr",
    ]
)
def gen_bitwise_or_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        assert len(inputs) > 1, "inputs must contain at least two elements"
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        if x0.dtype == torch.uint16:
            x0.numpy()
            x1.numpy()
            y = np.bitwise_or(x0, x1)
        else:
            y = torch.bitwise_or(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseOr", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseXor/BitwiseXorOperationTest.TestBitwiseXor",
    ]
)
def gen_bitwise_xor_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        assert len(inputs) > 1, "inputs must contain at least two elements"
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        if x0.dtype == torch.uint16:
            x0.numpy()
            x1.numpy()
            y = np.bitwise_xor(x0, x1)
        else:
            y = torch.bitwise_xor(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseXor", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMul/MulOperationTest.TestMul",
    ]
)
def gen_mul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] * inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Mul", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Maximum", "Minimum"])
def maximum_minimum_parameter(params: dict):
    params["scalar"] = 0 if not params.get("scalar") else params["scalar"]
    params["scalar_type"] = "" if not params.get("scalar_type") else params["scalar_type"]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMaximum/MaximumOperationTest.TestMaximum",
    ]
)
def gen_maximum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        is_element_mode = len(inputs) <= 1
        if is_element_mode:
            params = config["params"]
            x = inputs[0]
            scalar_type = params.get("scalar_type", "fp32")
            scalar = get_dtype_by_name(scalar_type)(params["scalar"])
            y = np.where(x < scalar, scalar, x)
            return [y]
        return [np.maximum(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Maximum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMinimum/MinimumOperationTest.TestMinimum",
    ]
)
def gen_minimum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        is_element_mode = len(inputs) <= 1
        if is_element_mode:
            params = config["params"]
            x = inputs[0]
            scalar_type = params.get("scalar_type", "fp32")
            scalar = get_dtype_by_name(scalar_type)(params["scalar"])
            y = np.where(x > scalar, scalar, x)
            return [y]
        return [np.minimum(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Minimum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDiv/DivOperationTest.TestDiv",
    ]
)
def gen_div_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        result = torch.div(from_numpy(inputs[0]), from_numpy(inputs[1]))
        return [to_numpy(result)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Div", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExpandExpDif/ExpandExpDifOperationTest.TestExpandExpDif",
    ]
)
def gen_expand_exp_dif_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        dtype_out = inputs[0].dtype
        dtype_in = np.float32 if dtype_out == bfloat16 else dtype_out
        return [np.exp(inputs[0].astype(dtype_in) - inputs[1].astype(dtype_in)).astype(dtype_out)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ExpandExpDif", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdds/AddsOperationTest.TestAdds",
    ]
)
def gen_adds_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] + params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Adds", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMuls/MulsOperationTest.TestMuls",
    ]
)
def gen_muls_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] * params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Muls", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDivs/DivsOperationTest.TestDivs",
    ]
)
def gen_divs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        result = torch.div(from_numpy(inputs[0]), params["scalar"])
        return [to_numpy(result)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Divs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFull/FullOperationTest.TestFull",
    ]
)
def gen_vector_dup_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [np.full(inputs[0].shape, params["scalar"], inputs[0].dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Full", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSubs/SubsOperationTest.TestSubs",
    ]
)
def gen_subs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] - params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Subs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseAnds/BitwiseAndsOperationTest.TestBitwiseAnds",
    ]
)
def gen_bitwise_ands_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "int16")
        numpy_scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        params["scalar"] = numpy_scalar.item()
        assert len(inputs) > 0, "inputs must contain at least one element"
        x = torch.tensor(inputs[0])
        if x.dtype == torch.uint16:
            x.numpy()
            y = np.bitwise_and(x, params["scalar"])
        else:
            y = torch.bitwise_and(x, params["scalar"])
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseAnds", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseOrs/BitwiseOrsOperationTest.TestBitwiseOrs",
    ]
)
def gen_bitwise_ors_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "int16")
        numpy_scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        params["scalar"] = numpy_scalar.item()
        assert len(inputs) > 0, "inputs must contain at least one element"
        x = torch.tensor(inputs[0])
        if x.dtype == torch.uint16:
            x.numpy()
            y = np.bitwise_or(x, params["scalar"])
        else:
            y = torch.bitwise_or(x, params["scalar"])
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseOrs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseXors/BitwiseXorsOperationTest.TestBitwiseXors",
    ]
)
def gen_bitwise_xors_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "int16")
        numpy_scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        params["scalar"] = numpy_scalar.item()
        assert len(inputs) > 0, "inputs must contain at least one element"
        x = torch.tensor(inputs[0])
        if x.dtype == torch.uint16:
            x.numpy()
            y = np.bitwise_xor(x, params["scalar"])
        else:
            y = torch.bitwise_xor(x, params["scalar"])
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseXors", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Sum", "Amax", "Amin", "Prod", "ArgMax", "ArgMin"])
def params_dims_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["keepDim"] = params.get("keepDim", True)
    assert isinstance(params["keepDim"], bool), "keepDim must be bool"
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSum/SumOperationTest.TestSum",
    ]
)
def gen_reduce_sum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = safe_tensor_conversion(inputs[0])
        input_dtype = x.dtype
        tensor_dtype = get_dtype_by_name(input_dtype, True)
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        y = x.sum(axis=dims[0], keepdims=keepdim)
        if input_dtype == bfloat16:
            y = y.to(torch.float32).numpy().astype(bfloat16)
        else:
            y = y.to(tensor_dtype).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAmax/AmaxOperationTest.TestAmax",
    ]
)
def gen_reduce_max_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.max(axis=dims[0], keepdims=keepdim)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Amax", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestArgMax/ArgMaxOperationTest.TestArgMax",
    ]
)
def gen_argmax_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.argmax(axis=dims[0], keepdims=keepdim).astype(np.int32)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ArgMax", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestArgMin/ArgMinOperationTest.TestArgMin",
    ]
)
def gen_argmin_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.argmin(axis=dims[0], keepdims=keepdim).astype(np.int32)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ArgMin", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAmin/AminOperationTest.TestAmin",
    ]
)
def gen_reduce_min_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [inputs[0].min(axis=dims[0], keepdims=keepdim)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Amin", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestProd/ProdOperationTest.TestProd",
    ]
)
def gen_reduce_prod_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.prod(axis=dims[0], keepdims=keepdim, dtype=inputs[0].dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Prod", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Transpose"])
def transpose_params_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["first_dim"] = int(params.pop("first_dim"))
    params["second_dim"] = int(params.pop("second_dim"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTranspose/TransposeOperationTest.TestTranspose",
    ]
)
def gen_transpose_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        return [np.transpose(inputs[0], axes=tuple(params["dims"]))]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Transpose", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["TransData"])
def permute_params_func(params: dict):
    params["valid_shape"] = parse_list_str(params.get("valid_shape"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTransData/TransDataOperationTest.TestTransData",
    ]
)
def gen_transpose_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    def golden_func(inputs: list, config: dict):
        if inputs[0].dtype == bfloat16:
            input_tensor = torch.as_tensor(inputs[0].astype(np.float32)).to(torch.bfloat16)
        else:
            input_tensor = torch.from_numpy(inputs[0])

        params = config.get("params")
        transdata_type = params["type"]
        group = params["group"]
        if transdata_type == 2:
            input_n, input_c, input_h, input_w = input_tensor.shape
            c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))
            per_group_c = input_c // group
            per_group_c1 = (per_group_c + c0 - 1) // c0
            total_c1 = per_group_c1 * group
            total_c = c0 * total_c1
            res = torch.zeros(input_n, total_c1, input_h, input_w, c0)
            for i in range(group):
                tmp_input_tensor = input_tensor[:, i * per_group_c:(i + 1) * per_group_c, :, :]
                if input_c < total_c:
                    pad_size = (total_c - input_c) // group
                    tmp_input_tensor = F.pad(tmp_input_tensor, (0, 0, 0, 0, 0, pad_size))
                tmp_input_tensor = tmp_input_tensor.view(input_n, per_group_c1, c0, input_h, input_w)
                tmp_res = tmp_input_tensor.permute(0, 1, 3, 4, 2).contiguous()
                res[:, i * per_group_c1:(i + 1) * per_group_c1, :, :, :] = tmp_res

            if inputs[0].dtype == bfloat16:
                res = res.to(torch.float32).numpy().astype(bfloat16)
                return [res]

            return [res.numpy()]
        elif transdata_type == 0:
            if len(input_tensor.shape) == 5 and len(config["output_tensors"][0]["shape"]) == 4:
                input_n, input_c1, input_h, input_w, c0 = input_tensor.shape
                total_c = c0 * input_c1
                output_c = config["output_tensors"][0]["shape"][1]
                res = torch.zeros(input_n, output_c, input_h, input_w)
                per_group_c1 = input_c1 // group
                per_group_c = per_group_c1 * c0
                per_output_group_c = output_c // group
                for i in range(group):
                    tmp_input_tensor = input_tensor[:, i * per_group_c1:(i + 1) * per_group_c1, :, :, :]
                    tmp_res = tmp_input_tensor.permute(0, 1, 4, 2, 3).reshape(input_n, per_group_c, input_h, input_w)
                    res[:, i * per_output_group_c:(i + 1) * per_output_group_c, :, :] = tmp_res[
                        :, :per_output_group_c, :, :
                    ]

                if inputs[0].dtype == bfloat16:
                    res = res.to(torch.float32).numpy().astype(bfloat16)
                    return [res]

                return [res.numpy()]
            elif len(input_tensor.shape) == 6:
                input_n, input_d, input_c1, input_h, input_w, c0 = input_tensor.shape
                total_c = c0 * input_c1
                output_c = config["output_tensors"][0]["shape"][1]
                res = torch.zeros(input_n, output_c, input_d, input_h, input_w)
                per_group_c1 = input_c1 // group
                per_group_c = per_group_c1 * c0
                per_output_group_c = output_c // group
                for i in range(group):
                    tmp_input_tensor = input_tensor[:, :, i * per_group_c1:(i + 1) * per_group_c1, :, :, :]
                    tmp_res = tmp_input_tensor.permute(0, 2, 5, 1, 3, 4).reshape(
                        input_n, per_group_c, input_d, input_h, input_w
                    )
                    res[:, i * per_output_group_c:(i + 1) * per_output_group_c, :, :, :] = tmp_res[
                        :, :per_output_group_c, :, :, :
                    ]

                if inputs[0].dtype == bfloat16:
                    res = res.to(torch.float32).numpy().astype(bfloat16)
                    return [res]

                return [res.numpy()]
            elif len(input_tensor.shape) == 4 and len(config["output_tensors"][0]["shape"]) == 4:
                input_c1hw, input_n1, input_n0, input_c0 = input_tensor.shape
                c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))
                n0 = 16

                output_n = config["output_tensors"][0]["shape"][0]
                output_per_n = output_n // group
                output_per_n1 = (output_per_n + n0 - 1) // n0
                output_c = config["output_tensors"][0]["shape"][1]
                output_c1 = (output_c + c0 - 1) // c0
                output_h = config["output_tensors"][0]["shape"][2]
                output_w = config["output_tensors"][0]["shape"][3]
                res = torch.zeros(output_n, output_c, output_h, output_w)

                output_pad_per_n = output_per_n1 * n0
                output_pad_per_c = output_c1 * c0
                intput_per_c1hw = input_c1hw // group

                for i in range(group):
                    tmp_input_tensor = input_tensor[i * intput_per_c1hw:(i + 1) * intput_per_c1hw, :, :, :]
                    tmp_res = tmp_input_tensor.reshape(
                        [output_c1, output_h, output_w, output_per_n1, n0, c0]).permute(
                            3, 4, 0, 5, 1, 2).reshape(
                                [output_pad_per_n, output_pad_per_c, output_h, output_w])

                    res[i * output_per_n:(i +1) * output_per_n:, :, :, :] = tmp_res[
                        0:output_per_n, 0:output_c, :, :
                    ]

                if inputs[0].dtype == bfloat16:
                    res = res.to(torch.float32).numpy().astype(bfloat16)
                    return [res]

                return [res.numpy()]
            elif len(input_tensor.shape) == 4 and len(config["output_tensors"][0]["shape"]) == 5:
                input_dc1hw, input_n1, input_n0, input_c0 = input_tensor.shape
                c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))
                n0 = 16

                output_n = config["output_tensors"][0]["shape"][0]
                output_per_n = output_n // group
                output_per_n1 = (output_per_n + n0 - 1) // n0
                output_c = config["output_tensors"][0]["shape"][1]
                output_c1 = (output_c + c0 - 1) // c0
                output_d = config["output_tensors"][0]["shape"][2]
                output_h = config["output_tensors"][0]["shape"][3]
                output_w = config["output_tensors"][0]["shape"][4]
                res = torch.zeros(output_n, output_c, output_d, output_h, output_w)

                output_pad_per_n = output_per_n1 * n0
                output_pad_per_c = output_c1 * c0
                input_per_dc1hw = input_dc1hw // group

                for i in range(group):
                    tmp_input_tensor = input_tensor[i * input_per_dc1hw:(i + 1) * input_per_dc1hw, :, :, :]
                    tmp_res = tmp_input_tensor.reshape(
                        [output_d, output_c1, output_h, output_w, output_per_n1, n0, c0]).permute(
                            4, 5, 1, 6, 0, 2, 3).reshape(
                                [output_pad_per_n, output_pad_per_c, output_d, output_h, output_w])

                    res[i * output_per_n:(i + 1) * output_per_n, :, :, :, :] = tmp_res[
                        0:output_per_n, 0:output_c, :, :, :
                    ]

                if inputs[0].dtype == bfloat16:
                    res = res.to(torch.float32).numpy().astype(bfloat16)
                    return [res]

                return [res.numpy()]
        elif transdata_type == 3:
            input_n, input_c, input_d, input_h, input_w = input_tensor.shape
            c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))

            per_group_c = input_c // group
            per_group_c1 = (per_group_c + c0 - 1) // c0
            total_c1 = per_group_c1 * group
            total_c = c0 * total_c1
            res = torch.zeros(input_n, input_d, total_c1, input_h, input_w, c0)
            for i in range(group):
                tmp_input_tensor = input_tensor[:, i * per_group_c:(i + 1) * per_group_c, :, :, :]
                if input_c < total_c:
                    pad_c_size = (total_c - input_c) // group
                    tmp_input_tensor = F.pad(tmp_input_tensor, (0, 0, 0, 0, 0, 0, 0, pad_c_size, 0, 0))
                tmp_res = tmp_input_tensor.view(input_n, per_group_c1, c0, input_d, input_h, input_w).permute(
                    0, 3, 1, 4, 5, 2
                )
                res[:, :, i * per_group_c1:(i + 1) * per_group_c1, :, :, :] = tmp_res

            if inputs[0].dtype == bfloat16:
                res = res.to(torch.float32).numpy().astype(bfloat16)
                return [res]

            return [res.numpy()]
        elif transdata_type == 4:
            input_n, input_c, input_h, input_w = input_tensor.shape
            c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))
            input_c1 = (input_c + c0 - 1) // c0
            n0 = 16
            per_group_n = input_n // group
            per_group_n1 = (per_group_n + n0 - 1) // n0
            total_c = c0 * input_c1
            res = torch.zeros(group * input_c1 * input_h * input_w, per_group_n1, n0, c0)
            for i in range(group):
                tmp_input_tensor = input_tensor[i * per_group_n:(i + 1) * per_group_n, :, :, :]
                if input_c < total_c or per_group_n < per_group_n1 * n0:
                    pad_c_size = total_c - input_c
                    pad_n_size = per_group_n1 * n0 - per_group_n
                    tmp_input_tensor = F.pad(tmp_input_tensor, (0, 0, 0, 0, 0, pad_c_size, 0, pad_n_size))
                tmp_input_tensor = tmp_input_tensor.view(per_group_n1, n0, input_c1, c0, input_h, input_w).permute(
                    2, 4, 5, 0, 1, 3
                )
                tmp_res = tmp_input_tensor.reshape([input_c1 * input_h * input_w, per_group_n1, n0, c0]).contiguous()
                res[i * input_c1 * input_h * input_w:(i + 1) * input_c1 * input_h * input_w, :, :, :] = tmp_res

            if inputs[0].dtype == bfloat16:
                res = res.to(torch.float32).numpy().astype(bfloat16)
                return [res]

            return [res.numpy()]
        elif transdata_type == 5:
            input_n, input_c, input_d, input_h, input_w = input_tensor.shape
            c0 = int(32 / get_c_sizeof_type(str(inputs[0].dtype)))
            input_c1 = (input_c + c0 - 1) // c0
            n0 = 16
            per_group_n = input_n // group
            per_group_n1 = (per_group_n + n0 - 1) // n0
            total_n1 = per_group_n1 * group
            total_n = total_n1 * n0
            total_c = c0 * input_c1
            res = torch.zeros(group * input_d * input_c1 * input_h * input_w, per_group_n1, n0, c0)
            for i in range(group):
                tmp_input_tensor = input_tensor[i * per_group_n:(i + 1) * per_group_n, :, :, :, :]
                if input_c < total_c or input_n < total_n:
                    pad_c_size = total_c - input_c
                    pad_n_size = (total_n - input_n) // group
                    tmp_input_tensor = F.pad(tmp_input_tensor, (0, 0, 0, 0, 0, 0, 0, pad_c_size, 0, pad_n_size))
                tmp_input_tensor = tmp_input_tensor.view(
                    per_group_n1, n0, input_c1, c0, input_d, input_h, input_w
                ).permute(4, 2, 5, 6, 0, 1, 3)
                tmp_res = tmp_input_tensor.reshape(
                    [input_d * input_c1 * input_h * input_w, per_group_n1, n0, c0]
                ).contiguous()
                res[
                    i * input_d * input_c1 * input_h * input_w:(i + 1) * input_d * input_c1 * input_h * input_w,
                    :,
                    :,
                    :,
                ] = tmp_res

            if inputs[0].dtype == bfloat16:
                res = res.to(torch.float32).numpy().astype(bfloat16)
                return [res]

            return [res.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("TransData", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Permute"])
def permute_params_func(params: dict):  # noqa: F811
    params["perm"] = parse_list_str(params.get("perm"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPermute/PermuteOperationTest.TestPermute",
    ]
)
def gen_permute_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        perm = tuple(params["perm"])
        input_arr = inputs[0]
        input_dtype_name = config["input_tensors"][0]["dtype"]
        if input_dtype_name == "hf8":
            y = np.transpose(input_arr, perm)
            return [y]
        if input_dtype_name == "bf16":
            x = safe_tensor_conversion(input_arr)
            y = torch.permute(x, dims=perm)
            y = y.to(torch.float32).numpy().astype(bfloat16)
            return [y]
        if input_dtype_name in ("fp8e4m3", "fp8e5m2", "fp8e8m0"):
            y = np.transpose(input_arr, perm)
            return [y.view(np.uint8) if y.dtype != np.uint8 else y]
        x = safe_tensor_conversion(input_arr)
        y = torch.permute(x, dims=perm)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Permute", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestWhere/WhereOperationTest.TestWhere",
    ]
)
def gen_where_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        condition_np = inputs[0]
        params = config.get("params")
        flag = params["flag"]
        x_scalar = params["x_scalar"]
        y_scalar = params["y_scalar"]

        def cast_condition(cond: np.ndarray) -> torch.Tensor:
            if cond.dtype == np.bool_:
                return torch.from_numpy(cond).bool()
            elif cond.dtype == np.uint8:
                orig_shape = cond.shape
                n = int(np.prod(orig_shape[:-1]))
                d = orig_shape[-1]
                reshaped = cond.reshape(n, d)
                bits = ((reshaped[:, :, None] >> np.arange(8)) & 1).astype(bool)
                expanded = bits.reshape(n, d * 8)
                new_shape = list(orig_shape[:-1]) + [d * 8]
                return torch.from_numpy(expanded.reshape(new_shape)).bool()
            else:
                raise TypeError(f"Unsupported condition dtype: {cond.dtype}")

        condition = cast_condition(condition_np)

        x = safe_tensor_conversion(inputs[1])
        y = safe_tensor_conversion(inputs[2])

        if flag == 0:
            res = torch.where(condition, x, y)
        elif flag == 1:
            res = torch.where(condition, x, y_scalar)
        elif flag == 2:
            res = torch.where(condition, x_scalar, y)
        elif flag == 3:
            res = torch.where(condition, x_scalar, y_scalar)
        else:
            raise ValueError(f"Invalid flag value: {flag}")

        if res.dtype == torch.bfloat16:
            res = res.to(torch.float32).numpy()
        else:
            res = res.numpy()
        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Where", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLReLU/LReLUOperationTest.TestLReLU",
    ]
)
def gen_leaky_relu_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        assert len(inputs) == 1, "LReLU expects exactly one input tensor"
        x = safe_tensor_conversion(inputs[0])

        params = config.get("params", {})
        scalar_val = params.get("scalar")
        if scalar_val is None:
            scalar_val = 0.01
        alpha = float(scalar_val)

        y = F.leaky_relu(x, negative_slope=alpha)
        if y.dtype == torch.bfloat16:
            y = y.to(torch.float32).numpy()
        else:
            y = y.numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("LReLU", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["TopK"])
def topk_params_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["count"] = parse_list_str(params.get("count"))
    params["islargest"] = [bool(x) for x in parse_list_str(params.get("islargest"))]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTopK/TopKOperationTest.TestTopK",
    ]
)
def gen_topk_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = torch.from_numpy(inputs[0])
        dims = params["dims"]
        count = params["count"]
        islargest = params["islargest"]
        val, idx = x.topk(count[0], dim=dims[0], largest=islargest[0], sorted=True)
        return [val.numpy(), idx.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("TopK", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Sort"])
def sort_params_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSort/SortOperationTest.TestSort",
    ]
)
def gen_sort_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = torch.from_numpy(inputs[0])
        dims = params["dims"]
        val, idx = torch.sort(x, dim=dims[0], descending=True, stable=True)
        return [val.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sort", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRange/RangeOperationTest.TestRange",
    ]
)
def gen_numpy_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        start = params["start"]
        end = params["end"]
        step = params["step"]
        output_tensors_type = config["output_tensors"][0]["dtype"]
        inputdata_type = get_dtype_by_name(output_tensors_type)
        if inputdata_type == bfloat16:
            result = torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float32)
            return [result.numpy().astype(bfloat16)]
        elif inputdata_type == np.int16:
            return [torch.arange(np.int16(start), np.int16(end), np.int16(step), dtype=torch.int16).numpy()]
        elif inputdata_type == np.float16:
            return [torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float16).numpy()]
        elif inputdata_type == np.float32:
            return [torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float32).numpy()]
        return [torch.arange(np.int32(start), np.int32(end), np.int32(step)).numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Range", golden_func, output, case_index)


class GatherError(Exception):
    pass


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGather/GatherOperationTest.TestGather",
    ]
)
def gen_gather_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        def _normalize_axis(axis, ndim):
            """把 axis 转成 [0, ndim) 区间"""
            if axis < 0:
                axis += ndim
            if not (0 <= axis < ndim):
                raise GatherError(f"axis={axis} 越界，ndim={ndim}")
            return axis

        def np_gather(params, indices, axis=0, batch_dims=0, *, validate_indices=True):
            """
            用 NumPy 模拟 TensorFlow 的 tf.gather。
            当 batch_dims==0 时直接走 np.take；当 batch_dims>0 时手动做 batch 切片再拼接。
            """
            p = np.asarray(params)
            idx = np.asarray(indices)

            # ---- 基础校验 ----
            if not np.issubdtype(idx.dtype, np.integer):
                raise GatherError("indices 必须是整数类型")
            if p.ndim == 0:
                raise GatherError("params 必须至少 1 维")

            axis = _normalize_axis(axis, p.ndim)

            # ---- batch_dims 归一化 ----
            if batch_dims < 0:
                batch_dims += min(p.ndim, idx.ndim)
            if not (0 <= batch_dims <= axis):
                raise GatherError("batch_dims 必须满足 0<=batch_dims<=axis")

            # ---- batch_dims == 0：直接 np.take ----
            if batch_dims == 0:
                if validate_indices:
                    size = p.shape[axis]
                    if idx.size > 0:
                        low, high = idx.min(), idx.max()
                        if low < -size or high >= size:
                            raise IndexError("indices 越界")
                mode = "raise" if validate_indices else "wrap"
                return np.take(p, idx, axis=axis, mode=mode)

            # ---- batch_dims > 0：逐 batch 切片 ----
            if idx.ndim < batch_dims:
                raise GatherError("indices 的秩必须 >= batch_dims")

            # 前 batch_dims 维必须完全对齐，否则按 TF 语义报错
            if p.shape[:batch_dims] != idx.shape[:batch_dims]:
                raise GatherError("params 与 indices 的前 batch_dims 维不一致")

            # 先做一次轴搬运，把 axis 移到 batch_dims 后面，方便后面统一处理
            move_to = batch_dims  # 目标位置
            if axis != move_to:
                p = np.moveaxis(p, axis, move_to)
                # 注意：搬完后真正的 轴 就是 move_to
                axis = move_to

            # 结果形状模板
            remain_dim = batch_dims + 1
            res_shape = (
                p.shape[:batch_dims]  # 保留的 batch 维
                + idx.shape[batch_dims:]  # indices 带来的新维
                + p.shape[remain_dim:]  # 剩下的数据维
            )

            # 提前分配结果
            res = np.empty(res_shape, dtype=p.dtype)

            # 遍历所有 batch 切片
            for b in np.ndindex(p.shape[:batch_dims]):
                # 取出当前 batch 的 params 切片，形状 (...,)  1D 轴
                p_slice = p[b]  # shape: [D_axis] + p.shape[batch_dims+1:]
                # 取出当前 batch 的 indices 切片
                idx_slice = idx[b]  # shape: idx.shape[batch_dims:]

                if validate_indices:
                    size = p_slice.shape[0]
                    if idx_slice.size > 0:
                        low, high = idx_slice.min(), idx_slice.max()
                        if low < -size or high >= size:
                            raise IndexError("indices 越界")

                mode = "raise" if validate_indices else "wrap"
                gathered = np.take(p_slice, idx_slice, axis=0, mode=mode)
                # 写回结果
                res[b] = gathered

            # 最后把 axis 搬回原始位置
            if move_to != axis:
                res = np.moveaxis(res, move_to, axis)
            return res

        params = config.get("params")
        axis = params["axis"]
        res = np_gather(inputs[0], inputs[1], axis)
        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Gather", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["GatherElement", "Concat", "Gather"])
def params_axis_func(params: dict):
    params["axis"] = int(params.get("axis"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGatherElement/GatherElementOperationTest.TestGatherElement",
    ]
)
def gen_gatherelement_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        axis = params["axis"]
        if inputs[1].dtype == np.int32:
            indices = torch.from_numpy(inputs[1]).long()
        else:
            indices = torch.from_numpy(inputs[1])
        if inputs[0].dtype == bfloat16:
            src = torch.from_numpy(inputs[0].astype(np.float32))
            res = src.gather(axis, indices).numpy().astype(bfloat16)
        else:
            src = torch.from_numpy(inputs[0])
            res = src.gather(axis, indices).numpy()

        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("GatherElement", golden_func, output, case_index)


def gcd_golden_func(inputs: list, config: dict):
    input = from_numpy(inputs[0])
    other = from_numpy(inputs[1])
    res = torch.gcd(input, other)
    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGcd/GcdOperationTest.TestGcd",
    ]
)
def gen_gcd_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Gcd", gcd_golden_func, output, case_index)


def gcds_golden_func(inputs: list, config: dict):
    input = from_numpy(inputs[0])
    params = config.get("params")
    other = int(params["scalar"])
    res = torch.gcd(input, torch.tensor(other))
    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGcds/GcdsOperationTest.TestGcds",
    ]
)
def gen_gcds_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Gcds", gcds_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGatherMask/GatherMaskOperationTest.TestGatherMask",
    ]
)
def gen_gathermask_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        orig_dtype = inputs[0].dtype
        if orig_dtype == np.uint32:
            input_tensor = torch.from_numpy(inputs[0].astype(np.int32))
        elif orig_dtype == np.uint16:
            input_tensor = torch.from_numpy(inputs[0].astype(np.int16))
        else:
            input_tensor = torch.from_numpy(inputs[0])
        params = config.get("params")
        pattern_mode = int(params.get("patternMode"))
        last_dim = input_tensor.shape[-1]
        if last_dim % 2 != 0 and pattern_mode in [1, 2]:
            raise ValueError("The last axis should be divisible by 2 when ptternMode is 1 or 2")
        if last_dim % 4 != 0 and pattern_mode in [3, 4, 5, 6]:
            raise ValueError("The last axis should be divisible by 4 when ptternMode is 3,4,5 or 6")
        # 获取索引
        indices = torch.arange(last_dim)

        if pattern_mode == 7:
            return [inputs[0]]
        else:
            if pattern_mode == 1:
                # 每两个取第一个
                selected_indices = indices[::2]
            elif pattern_mode == 2:
                # 每两个取第二个
                selected_indices = indices[1::2]
            elif pattern_mode == 3:
                # 每四个取第一个
                selected_indices = indices[::4]
            elif pattern_mode == 4:
                # 每四个取第二个
                selected_indices = indices[1::4]
            elif pattern_mode == 5:
                # 每四个取第三个
                selected_indices = indices[2::4]
            elif pattern_mode == 6:
                # 每四个取第四个
                selected_indices = indices[3::4]
            # 使用索引选择元素
            output = input_tensor.index_select(-1, selected_indices)
            return [output.numpy().astype(orig_dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("GatherMask", golden_func, output, case_index)


def cumsum_golden_func(inputs: list, config: dict):
    params = config.get("params")
    axis = params["axis"]
    if inputs[0].dtype == bfloat16:
        input_tensor = torch.as_tensor(inputs[0].astype(np.float32)).to(torch.bfloat16)
    else:
        input_tensor = torch.from_numpy(inputs[0])
    res = torch.cumsum(input_tensor, axis)
    if inputs[0].dtype == bfloat16:
        res = res.to(torch.float32).numpy().astype(bfloat16)
        return [res]

    return [res.numpy()]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCumSum/CumSumOperationTest.TestCumSum",
    ]
)
def gen_cumsum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("CumSum", cumsum_golden_func, output, case_index)


def cumprod_golden_func(inputs: list, config: dict):
    params = config.get("params")
    axis = params["axis"]
    if inputs[0].dtype == bfloat16:
        input_tensor = torch.as_tensor(inputs[0].astype(np.float32)).to(torch.bfloat16)
    else:
        input_tensor = torch.from_numpy(inputs[0])
    res = torch.cumprod(input_tensor, axis)
    if inputs[0].dtype == bfloat16:
        res = res.to(torch.float32).numpy().astype(bfloat16)
        return [res]

    return [res.numpy()]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCumProd/CumProdOperationTest.TestCumProd",
    ]
)
def gen_cumprod_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("CumProd", cumprod_golden_func, output, case_index)


def triu_golden_func(inputs: list, config: dict):
    params = config.get("params")
    diagonal = params["diagonal"]
    input_tensor = from_numpy(inputs[0])
    res = torch.triu(input_tensor, diagonal)
    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTriU/TriUOperationTest.TestTriU",
    ]
)
def gen_triu_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("TriU", triu_golden_func, output, case_index)


def tril_golden_func(inputs: list, config: dict):
    params = config.get("params")
    diagonal = params["diagonal"]
    input_tensor = from_numpy(inputs[0])
    res = torch.tril(input_tensor, diagonal)
    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTriL/TriLOperationTest.TestTriL",
    ]
)
def gen_tril_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("TriL", tril_golden_func, output, case_index)


def from_numpy(array: np.array):
    is_bfloat16 = array.dtype == bfloat16
    if is_bfloat16:
        result = torch.tensor(array.astype(np.float32), dtype=torch.bfloat16)
    else:
        result = torch.tensor(array)
    return result


def to_numpy(tensor: torch.Tensor):
    is_bfloat16 = tensor.dtype == torch.bfloat16
    if is_bfloat16:
        result = tensor.to(torch.float32).numpy().astype(bfloat16)
    else:
        result = tensor.numpy()
    return result


def indexadd_golden_func(inputs: list, config: dict):
    params = config.get("params")
    axis = params["axis"]
    self = from_numpy(inputs[0])
    source = from_numpy(inputs[1])
    indices = from_numpy(inputs[2])
    try:
        alp = float(params["alpha"])
    except (KeyError, ValueError, TypeError):
        alp = 1
    res = self.index_add(axis, indices, source, alpha=alp)

    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestIndexAddUB/IndexAddUBOperationTest.TestIndexAddUB",
    ]
)
def gen_indexadd_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("IndexAddUB", indexadd_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestIndexAdd_/IndexAdd_OperationTest.TestIndexAdd_",
    ]
)
def gen_indexadd__op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("IndexAdd_", indexadd_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(case_names=["TestRemainder/RemainderOperationTest.TestRemainder"])
def gen_remainder_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)

    def remainder_golden_func(inputs: list, config: dict):
        self = from_numpy(inputs[0])
        other = from_numpy(inputs[1])
        res = torch.remainder(self, other)
        return [to_numpy(res)]

    return gen_op_golden("Remainder", remainder_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(case_names=["TestRemainderS/RemainderSOperationTest.TestRemainderS"])
def gen_remainders_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)

    def remainders_golden_func(inputs: list, config: dict):
        params = config.get("params")
        other = params["scalar"]
        self = from_numpy(inputs[0])
        res = torch.remainder(self, other)
        return [to_numpy(res)]

    return gen_op_golden("RemainderS", remainders_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(case_names=["TestRemainderRS/RemainderRSOperationTest.TestRemainderRS"])
def gen_remainderrs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)

    def remainders_golden_func(inputs: list, config: dict):
        params = config.get("params")
        self = params["scalar"]
        other = from_numpy(inputs[0])
        res = torch.remainder(self, other)
        return [to_numpy(res)]

    return gen_op_golden("RemainderRS", remainders_golden_func, output, case_index)


def indexput_dfs(indices_range, deep, max_count, cur_indices, all_indices):
    if deep == max_count:
        all_indices.append([i for i in cur_indices])
        return
    for i in range(indices_range[deep][0], indices_range[deep][1]):
        cur_indices[deep] = i
        indexput_dfs(indices_range, deep + 1, max_count, cur_indices, all_indices)


def indexput_golden_func(indexput_config):
    output_path = indexput_config["output_path"]
    indices_range = indexput_config["indices_range"]
    dtype = indexput_config["dtype"]
    indices_dtype = indexput_config["indices_dtype"]
    indices_count = len(indices_range)
    input_path = Path(output_path, 'input0.bin')
    values_path = Path(output_path, 'input1.bin')
    indices_paths = [Path(output_path, f'input{2 + i}.bin') for i in range(indices_count)]
    input_ = np.random.uniform(
        indexput_config["input_range"][0], indexput_config["input_range"][1], size=indexput_config["input_shape"]
    ).astype(dtype)
    input_.tofile(input_path)
    values = np.random.uniform(
        indexput_config["values_range"][0], indexput_config["values_range"][1], size=indexput_config["values_shape"]
    ).astype(dtype)
    values.tofile(values_path)
    result = 1
    for arr in indices_range:
        result *= 1 if arr[0] == arr[1] else arr[1] - arr[0]
    indices_shape = indexput_config["values_shape"][0]
    indices = np.zeros((indices_count, indices_shape), dtype=indices_dtype)
    if indices_shape < result // 2:
        cnt = 0
        visit = set()
        while cnt < indices_shape:
            tmp = [0 for i in range(indices_count)]
            for i in range(indices_count):
                tmp[i] = random.randrange(indices_range[i][0], indices_range[i][1])
            if tuple(tmp) in visit:
                continue
            visit.add(tuple(tmp))
            for i in range(indices_count):
                indices[i][cnt] = tmp[i]
            cnt += 1
        for i in range(indices_count):
            indices[i].tofile(indices_paths[i])
    else:
        cur_indices = [0 for i in range(indices_count)]
        all_indices = []
        indexput_dfs(indices_range, 0, indices_count, cur_indices, all_indices)
        perm = np.random.permutation(len(all_indices))
        for i in range(indices_count):
            for j in range(indices_shape):
                indices[i][j] = all_indices[perm[j]][i]
            indices[i].tofile(indices_paths[i])
    result_path = Path(output_path, 'output0.bin')
    if indexput_config["accumulate"]:
        input_[tuple(indices)] += values
    else:
        input_[tuple(indices)] = values
    input_.tofile(result_path)
    return True


def indexput_pre_golden_func(output_path: Path, config: dict):
    input_tensors = config["input_tensors"]
    accumulate = config["params"]["accumulate"]
    input_ = input_tensors[0]
    values = input_tensors[1]
    input_data_range = input_["data_range"]
    values_data_range = values["data_range"]
    indices_range = []
    for i in range(len(input_tensors) - 2):
        indices = input_tensors[i + 2]
        indices_data_range = indices["data_range"]
        indices_range.append([indices_data_range['min'], indices_data_range['max']])
    indexput_config = {
        "output_path": output_path,
        "input_shape": input_["shape"],
        "values_shape": values["shape"],
        "input_range": [input_data_range["min"], input_data_range["max"]],
        "values_range": [values_data_range["min"], values_data_range["max"]],
        "indices_range": indices_range,
        "dtype": get_dtype_by_name(input_["dtype"]),
        "indices_dtype": get_dtype_by_name(input_tensors[2]["dtype"]),
        "accumulate": accumulate,
    }
    return indexput_golden_func(indexput_config)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestIndexPut_/IndexPut_OperationTest.TestIndexPut_",
    ]
)
def gen_indexput__op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    case_file: Path = Path(Path(__file__).parent.parent, "test_case/IndexPut__st_test_cases.json").resolve()
    test_configs = load_test_cases_from_json(str(case_file))
    if len(test_configs) == 0:
        raise ValueError("Not find test cases, please check.")
    return indexput_pre_golden_func(output, test_configs[case_index])


@TestCaseLoader.reg_params_handler(ops=["Scatter", "ScatterTensor"])
def params_axis_reduce_func(params: dict):
    params["axis"] = int(params.get("axis"))
    params["reduce"] = "" if params["reduce"] is None else params["reduce"]
    return params


def scatter_golden_func(inputs, config: dict):
    params = config.get("params")
    axis = params["axis"]
    reduceop = params["reduce"]
    scalar = params["src"]
    if inputs[1].dtype == np.int32:
        indices = torch.from_numpy(inputs[1]).long()
    else:
        indices = torch.from_numpy(inputs[1])

    if inputs[0].dtype == bfloat16:
        bf16_scalar = np.array([scalar], np.float32).astype(inputs[0].dtype).astype(np.float32)
        src = torch.from_numpy(inputs[0].astype(np.float32))
        if len(reduceop) == 0 or reduceop == "None":
            res = src.scatter(axis, indices, bf16_scalar[0]).numpy().astype(inputs[0].dtype)
        else:
            res = src.scatter(axis, indices, bf16_scalar[0], reduce=reduceop).numpy().astype(inputs[0].dtype)
    elif np.issubdtype(inputs[0].dtype, np.integer):
        src = torch.from_numpy(inputs[0])
        int_scalar = int(np.dtype(inputs[0].dtype).type(scalar))
        if len(reduceop) == 0 or reduceop == "None":
            res = src.scatter(axis, indices, int_scalar).numpy()
        else:
            res = src.scatter(axis, indices, int_scalar, reduce=reduceop).numpy()
    else:
        src = torch.from_numpy(inputs[0])
        if len(reduceop) == 0 or reduceop == "None":
            res = src.scatter(axis, indices, scalar).numpy()
        else:
            res = src.scatter(axis, indices, scalar, reduce=reduceop).numpy()

    return [res]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatter/ScatterOperationTest.TestScatter",
    ]
)
def gen_scatter_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Scatter", scatter_golden_func, output, case_index)


def scatter_tensor_golden_func(inputs, config: dict):
    params = config.get("params")
    axis = params["axis"]
    reduceop = params["reduce"]

    input1_tensor = config["input_tensors"][1]
    data_min = input1_tensor["data_range"]["min"]
    data_max = input1_tensor["data_range"]["max"]
    shape = inputs[1].shape
    dtype = inputs[1].dtype
    dims = inputs[1].ndim
    # 当前A5 vscatter指令在index索引重复的情况下并不保证，计算时序，生成golden时需要保证index数据不重复。
    # 仅axis为尾轴时，会出现此问题，因为pto内部处理时，单次只处理index的一行数据，axis为其他值时单次处理并不会导致
    # 其他场景，例如reduce为add，当前还是用标量场景
    is_regen_index = (
        (axis == dims - 1 or axis + dims == dims - 1)
        and (len(reduceop) == 0 or reduceop == "None")
        and (data_max >= shape[-1])
    )
    if is_regen_index:
        if dims == 2:
            for i in range(shape[0]):
                inputs[1][i] = np.random.choice(range(data_min, data_max), shape[-1], False).astype(dtype)
        elif dims == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    inputs[1][i, j] = np.random.choice(range(data_min, data_max), shape[-1], False).astype(dtype)
        elif dims == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        inputs[1][i, j, k] = np.random.choice(range(data_min, data_max), shape[-1], False).astype(dtype)
        else:
            raise ValueError("Dims is not supported, please check.")

    if inputs[1].dtype == np.int32:
        indices = torch.from_numpy(inputs[1]).long()
    else:
        indices = torch.from_numpy(inputs[1])

    if inputs[0].dtype == bfloat16:
        dst = torch.from_numpy(inputs[0].astype(np.float32))
        src = torch.from_numpy(inputs[2].astype(np.float32))
        if len(reduceop) == 0 or reduceop == "None":
            res = dst.scatter(axis, indices, src).numpy().astype(inputs[0].dtype)
        else:
            res = dst.scatter(axis, indices, src, reduce=reduceop).numpy().astype(inputs[0].dtype)
    else:
        dst = torch.from_numpy(inputs[0])
        src = torch.from_numpy(inputs[2])
        if len(reduceop) == 0 or reduceop == "None":
            res = dst.scatter(axis, indices, src).numpy()
        else:
            res = dst.scatter(axis, indices, src, reduce=reduceop).numpy()

    return [res]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatterTensor/ScatterTensorOperationTest.TestScatterTensor",
    ]
)
def gen_scatter_tensor_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ScatterTensor", scatter_tensor_golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Var"])
def params_dim_correction_keepdim_func(params: dict):
    params["dim"] = [] if params.get("dim") is None else parse_list_str(str(params.get("dim")))
    params["correction"] = float(params.get("correction"))
    params["keepDim"] = params.get("keepDim")
    assert isinstance(params["keepDim"], bool), "keepDim must be bool"
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestVar/VarOperationTest.TestVar",
    ]
)
def gen_var_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs, config: dict):
        params = config.get("params")
        dim_list = params["dim"]
        correction = params["correction"]

        if inputs[0].dtype == bfloat16:
            input = torch.from_numpy(inputs[0].astype(np.float32))
        else:
            input = torch.from_numpy(inputs[0])

        res = prims.var(input, dim_list, correction)

        return [res.numpy().astype(inputs[0].dtype)]

    return gen_op_golden("Var", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestConcat/ConcatOperationTest.TestConcat",
    ]
)
def gen_concat_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs, config: dict):
        params = config.get("params")
        axis = params["axis"]
        output_tensors_type = config["output_tensors"][0]["dtype"]
        inputdata_type = get_dtype_by_name(output_tensors_type)
        if inputdata_type == bfloat16:
            inputs_tensors = [torch.as_tensor(x.astype(np.float32)).to(torch.bfloat16) for x in inputs]
        else:
            inputs_tensors = [torch.as_tensor(x) for x in inputs]
        res = torch.cat(inputs_tensors, dim=axis)
        if inputdata_type == bfloat16:
            res = res.to(torch.float32).numpy().astype(bfloat16)
            return [res]
        return [res.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Concat", golden_func, output, case_index)


def _to_interleave_torch_tensor(array: np.ndarray) -> torch.Tensor:
    if array.dtype == bfloat16:
        return torch.as_tensor(array.astype(np.float32)).to(torch.bfloat16)
    return torch.as_tensor(array)


def _torch_interleave(src0: torch.Tensor, src1: torch.Tensor):
    interleaved = torch.stack((src0, src1), dim=-1).flatten(-2)
    last_dim = src0.shape[-1]
    return interleaved[..., :last_dim], interleaved[..., last_dim:]


def _torch_deinterleave(inputs: list):
    if len(inputs) == 1:
        interleaved = inputs[0]
    else:
        interleaved = torch.cat((inputs[0], inputs[1]), dim=-1)
    return interleaved[..., 0::2], interleaved[..., 1::2]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestInterleave/InterleaveOperationTest.TestInterleave",
    ]
)
def gen_interleave_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, _config: dict):
        src0 = _to_interleave_torch_tensor(inputs[0])
        src1 = _to_interleave_torch_tensor(inputs[1])
        out0, out1 = _torch_interleave(src0, src1)
        return [to_numpy(out0), to_numpy(out1)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Interleave", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDeInterleave/DeInterleaveOperationTest.TestDeInterleave",
    ]
)
def gen_deinterleave_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, _config: dict):
        torch_inputs = [_to_interleave_torch_tensor(input_data) for input_data in inputs]
        out0, out1 = _torch_deinterleave(torch_inputs)
        return [to_numpy(out0), to_numpy(out1)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("DeInterleave", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCompare/CompareOperationTest.TestCompare",
        "TestCompareBitMode/CompareOperationTest.TestCompareBitMode",
        "TestCmps/CmpsOperationTest.TestCmps",
        "TestCmpsBitMode/CmpsOperationTest.TestCmpsBitMode",
    ]
)
def gen_compare_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params", {})
        operation = params["compare_op"]
        mode = params.get("mode", "bool")
        input1 = safe_tensor_conversion(inputs[0])
        if len(inputs) > 1:
            input2 = safe_tensor_conversion(inputs[1])
        elif "scalar" in params:
            input2 = params["scalar"]
        else:
            input2 = params.get("other", 0)
        cmp_operations = {
            "eq": torch.eq,
            "ne": torch.ne,
            "lt": torch.lt,
            "le": torch.le,
            "gt": torch.gt,
            "ge": torch.ge,
        }
        result = cmp_operations[operation](input1, input2)
        if mode == "bit":
            bool_np = result.numpy()

            last_dim = bool_np.shape[-1] if bool_np.shape else 0
            if last_dim % 8 != 0:
                raise ValueError(f"Last dimension {last_dim} must be divisible by 8 in BIT mode")
            new_shape = list(bool_np.shape[:-1]) + [last_dim // 8, 8]
            bool_reshaped = bool_np.reshape(new_shape)
            bitmask = np.packbits(bool_reshaped, axis=-1, bitorder="little")
            bitmask = bitmask.reshape(bool_np.shape[:-1] + (last_dim // 8,))
            result = torch.tensor(bitmask, dtype=torch.uint8)
        else:
            result = result.to(torch.bool)
        return [result.numpy()]

    op_type = "Compare" if "Compare" in case_name else "Cmps"
    logging.debug("Case(%s), %s Golden creating...", case_name, op_type)
    return gen_op_golden(op_type, golden_func, output, case_index)


def as_float(value):
    """将数字、inf、nan、-inf 等字符串转换为浮点数"""
    if isinstance(value, str):
        value = value.lower()
        if not value:
            return None
    value = float(value)
    return value


def uniform_golden_func(inputs: list, config: dict):
    params = config.get("params", {})
    rounds = params.get("rounds", 10)
    if isinstance(rounds, str):
        rounds = int(rounds)
    shape = config["output_tensors"][0]["shape"]
    output_dtype = config["output_tensors"][0].get("dtype", "fp32").lower()

    key = params.get("key", 0)
    if isinstance(key, str):
        key = int(key)

    counter_1 = params.get("counter_1", 0)
    if isinstance(counter_1, str):
        counter_1 = int(counter_1)
    counter = [0, counter_1]

    def uint32(x):
        return x & 0xFFFFFFFF

    def multiply_high_low(a, b):
        product = a * b
        hi = uint32(product >> 32)
        lo = uint32(product)
        return lo, hi

    def philox_single_round(counter, key0, key1):
        lo0, hi0 = multiply_high_low(0xD2511F53, counter[0])
        lo1, hi1 = multiply_high_low(0xCD9E8D57, counter[2])

        return [uint32(hi1 ^ counter[1] ^ key0), uint32(lo1), uint32(hi0 ^ counter[3] ^ key1), uint32(lo0)]

    def raise_key(key0, key1):
        return (uint32(key0 + 0x9E3779B9), uint32(key1 + 0xBB67AE85))

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    result = np.zeros(total_elements, dtype=np.uint32)

    init_key0 = uint32(key)
    init_key1 = uint32(key >> 32)

    original_counter = [uint32(counter[0]), uint32(counter[0] >> 32), uint32(counter[1]), uint32(counter[1] >> 32)]

    for i in range(0, total_elements, 4):
        key0, key1 = init_key0, init_key1
        current_counter = original_counter.copy()

        for _ in range(rounds):
            current_counter = philox_single_round(current_counter, key0, key1)
            key0, key1 = raise_key(key0, key1)

        for j in range(min(4, total_elements - i)):
            result[i + j] = current_counter[j]

        original_counter[0] = uint32(original_counter[0] + 1)
        if original_counter[0] == 0:
            original_counter[1] = uint32(original_counter[1] + 1)
            if original_counter[1] == 0:
                original_counter[2] = uint32(original_counter[2] + 1)
                if original_counter[2] == 0:
                    original_counter[3] = uint32(original_counter[3] + 1)

    if output_dtype == "fp32":
        result_float = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            x = result[i]
            man = x & 0x7FFFFF
            exp = 127
            val = (exp << 23) | man
            result_float[i] = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0] - 1.0
        return [result_float.reshape(shape)]
    elif output_dtype == "fp16":
        result_half = np.zeros(total_elements, dtype=np.float16)
        for i in range(total_elements):
            x = result[i]
            x_uint16 = np.uint16(x & 0xFFFF)
            man = x_uint16 & 0x3FF
            exp = np.uint16(15)
            val = (exp << 10) | man
            result_half[i] = np.frombuffer(np.array([val], dtype=np.uint16).tobytes(), dtype=np.float16)[
                0
            ] - np.float16(1.0)
        return [result_half.reshape(shape)]
    elif output_dtype == "bf16":
        result_bfloat16 = np.zeros(total_elements, dtype=bfloat16)
        for i in range(total_elements):
            x = result[i]
            x_uint16 = np.uint16(x & 0xFFFF)
            man = x_uint16 & 0x7F
            exp = np.uint16(127)
            val = (exp << 7) | man
            result_bfloat16[i] = np.frombuffer(np.array([val], dtype=np.uint16).tobytes(), dtype=bfloat16)[
                0
            ] - bfloat16(1.0)
        return [result_bfloat16.reshape(shape)]
    else:
        result_float = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            x = result[i]
            man = x & 0x7FFFFF
            exp = 127
            val = (exp << 23) | man
            result_float[i] = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0] - 1.0
        return [result_float.reshape(shape)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestUniform/UniformOperationTest.TestUniform",
    ]
)
def gen_uniform_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Uniform", uniform_golden_func, output, case_index)


def safe_tensor_conversion(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == bfloat16:
        return torch.tensor(arr.astype(np.float32), dtype=torch.bfloat16)
    else:
        return torch.tensor(arr)


def element_mode(inputs, params, is_bfloat16):
    test_type = int(params["test_type"])
    min_dtype_str, max_dtype_str = params.get("min_dtype", ""), params.get("max_dtype", "")
    if min_dtype_str and test_type in [-1, 0, 1, 2, 7]:
        min_dtype = get_dtype_by_name(min_dtype_str, is_torch=True)
        min_value = as_float(params.get("min_value"))
        if min_value is not None:
            min_ = torch.tensor(min_value, dtype=min_dtype)
        else:
            min_ = None
    else:
        min_ = None
    if max_dtype_str and test_type in [-1, 0, 1, 2, 8]:
        max_dtype = get_dtype_by_name(max_dtype_str, is_torch=True)
        max_value = as_float(params.get("max_value"))
        if max_value is not None:
            max_ = torch.tensor(max_value, dtype=max_dtype)
        else:
            max_ = None
    else:
        max_ = None
    return min_, max_


def tensor_mode(inputs, params, is_bfloat16):
    test_type = int(params["test_type"])

    if test_type in [-1, 0, 1, 2, 5]:
        min_ = (
            torch.tensor(inputs[1].astype(np.float32), dtype=torch.bfloat16) if is_bfloat16 else torch.tensor(inputs[1])
        )
    else:
        min_ = None
    if test_type in [-1, 0, 1, 2, 4]:
        max_ = (
            torch.tensor(inputs[2].astype(np.float32), dtype=torch.bfloat16) if is_bfloat16 else torch.tensor(inputs[2])
        )
    else:
        max_ = None
    return min_, max_


@TestCaseLoader.reg_params_handler(ops=["Clip"])
def clip_parameter(params: dict):
    params["max_value"] = "" if not params.get("max_value") else params["max_value"]
    params["min_value"] = "" if not params.get("min_value") else params["min_value"]
    params["max_dtype"] = "" if not params.get("min_dtype") else params["min_dtype"]
    params["min_dtype"] = "" if not params.get("min_dtype") else params["min_dtype"]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestClip/ClipOperationTest.TestClip",
    ]
)
def gen_clip_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params", {})
        is_bfloat16 = inputs[0].dtype == bfloat16
        x = torch.tensor(inputs[0].astype(np.float32), dtype=torch.bfloat16) if is_bfloat16 else torch.tensor(inputs[0])
        min_dtype_str, max_dtype_str = params.get("min_dtype", ""), params.get("max_dtype", "")
        is_element = min_dtype_str or max_dtype_str
        if is_element:
            min_, max_ = element_mode(inputs, params, is_bfloat16)
        else:
            min_, max_ = tensor_mode(inputs, params, is_bfloat16)
        # Torch 不支持两者均为 None
        if min_ is None and max_ is None:
            min_ = float("-inf")
        result = torch.clip(x, min_, max_)
        return [(result.to(torch.float32).numpy().astype(bfloat16) if is_bfloat16 else result.numpy())]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Clip", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["ArgSort"])
def topk_params_func(params: dict):  # noqa: F811
    params["dims"] = parse_list_str(params.get("dims"))
    params["descending"] = [bool(x) for x in parse_list_str(params.get("descending"))]
    return params


@GoldenRegister.reg_golden_func(case_names=['TestArgSort/ArgSortOperationTest.TestArgSort'])
def gen_argsort_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = torch.from_numpy(inputs[0])
        dims = params["dims"]
        descending = params["descending"]
        idx = torch.argsort(x, dim=dims[0], descending=descending[0], stable=True)
        return [idx.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ArgSort", golden_func, output, case_index)


def _decode_e4m3_fn(code: int) -> float:
    sign = -1 if (code & 0x80) != 0 else 1
    exp = (code >> 3) & 0x0F
    mant = code & 0x07
    if exp == 0:
        if mant == 0:
            return -0.0 if sign < 0 else 0.0
        return float(sign) * math.ldexp(float(mant), -9)
    if exp == 0x0F and mant == 0x07:
        return math.nan
    significand = 1.0 + float(mant) / 8.0
    return float(sign) * math.ldexp(significand, exp - 7)


# MX quantization constants per target dtype (OCP Microscaling Formats MX v1.0).
_MX_DTYPE_PARAMS = {
    "fp8_e4m3": {
        "target_max_pow2": 8,
        "max_pos": 448.0,
        "min_normal": 2 ** (1 - 7),  # 2^-6 = 0.015625
        "exp_bias": 7,
        "mbits": 3,
    },
    "fp4_e2m1x2": {
        "target_max_pow2": 2,
        "max_pos": 6.0,
        "min_normal": 1.0,
        "exp_bias": 1,
        "mbits": 1,
    },
}

_QUANTMX_OUTPUT_TO_DTYPE = {
    "fp8e4m3": "fp8_e4m3",
    "fp4_e2m1x2": "fp4_e2m1x2",
}

_E8M0_EXPONENT_BIAS = 127
_F32_EXP_BIAS = 127
_F32_MBITS = 23


def _compute_shared_exponents_floor(max_abs: np.ndarray, target_max_pow2: int) -> np.ndarray:
    """Vectorized OCP FLOOR-mode shared exponent computation.

    Returns an ndarray of E8M0 biased bytes (uint8).
    Reference: OCP MX Spec 1.0 — scale = 2^floor(log2(max_abs)) / 2^target_max_pow2
    """
    nan_mask = np.isnan(max_abs)
    bits = max_abs.view(np.int32)
    fp_exponent = ((bits >> _F32_MBITS) & 0xFF).astype(np.int32)
    biased = np.clip(fp_exponent - target_max_pow2, 0, 254).astype(np.uint8)
    # Match torchao FLOOR mode: Inf follows the exponent path, while NaN maps to E8M0 NaN.
    biased[nan_mask] = 0xFF
    return biased


def _compute_shared_exponents_nv(max_abs: np.ndarray, max_pos: float) -> np.ndarray:
    """NV MX shared exponent computation using high precision math."""
    max_abs_ld = np.asarray(max_abs, dtype=np.longdouble)
    result = np.zeros(max_abs.shape, dtype=np.uint8)
    nan_mask = np.isnan(max_abs_ld)
    result[nan_mask] = 0xFF

    with np.errstate(divide="ignore", invalid="ignore"):
        descale = max_abs_ld / np.longdouble(max_pos)
    inf_mask = np.isinf(descale) & (~nan_mask)
    result[inf_mask] = 0xFE

    threshold = np.longdouble(2.0) ** np.longdouble(-_E8M0_EXPONENT_BIAS)
    active_mask = (~nan_mask) & (~inf_mask) & (descale > threshold)
    if np.any(active_mask):
        exponents = np.ceil(np.log2(descale[active_mask])) + np.longdouble(_E8M0_EXPONENT_BIAS)
        result[active_mask] = np.clip(exponents, 0, 254).astype(np.uint8)
    return result


def _compute_shared_exponents_ocp_math(max_abs: np.ndarray, target_max_pow2: int) -> np.ndarray:
    """OCP shared exponent computation using high precision math for FP4 golden."""
    max_abs_ld = np.asarray(max_abs, dtype=np.longdouble)
    result = np.zeros(max_abs.shape, dtype=np.uint8)
    nan_mask = np.isnan(max_abs_ld)
    result[nan_mask] = 0xFF
    inf_mask = np.isinf(max_abs_ld) & (~nan_mask)
    # Match torchao FLOOR/OCP: Inf follows the exponent path instead of NV satfinite.
    result[inf_mask] = np.uint8(max(0, min(0xFE, 0xFF - target_max_pow2)))

    threshold = np.longdouble(2.0) ** np.longdouble(target_max_pow2 - _E8M0_EXPONENT_BIAS)
    active_mask = (~nan_mask) & (~inf_mask) & (max_abs_ld > threshold)
    if np.any(active_mask):
        exponents = (
            np.floor(np.log2(max_abs_ld[active_mask]))
            - np.longdouble(target_max_pow2)
            + np.longdouble(_E8M0_EXPONENT_BIAS)
        )
        result[active_mask] = np.clip(exponents, 0, 254).astype(np.uint8)
    return result


_QUANTMX_SUPPORTED_MODES = {"ROUND_DOWN", "ROUND_UP"}


def _compute_scalings_from_exponents(e8m0: np.ndarray) -> np.ndarray:
    """Vectorized reciprocal scaling factor from E8M0 biased exponents.

    reciprocal_scale = 2^(E8M0_BIAS - e8m0) so that data * reciprocal_scale = data / scale.
    """
    e8m0_i32 = e8m0.astype(np.int32)
    scale_exp = np.int32(254) - e8m0_i32
    result = (scale_exp << _F32_MBITS).astype(np.int32).view(np.float32)
    result[scale_exp == 0] = np.float32(math.ldexp(1.0, -_E8M0_EXPONENT_BIAS))
    result[e8m0 == 0xFF] = np.float32(np.nan)
    return result


def _compute_scalings_from_exponents_math(e8m0: np.ndarray) -> np.ndarray:
    """Reciprocal scaling factor from E8M0 using math instead of bit construction."""
    e8m0_i32 = e8m0.astype(np.int32)
    result = np.ldexp(np.ones(e8m0.shape, dtype=np.float64), _E8M0_EXPONENT_BIAS - e8m0_i32).astype(np.float32)
    result[e8m0 == 0xFF] = np.float32(np.nan)
    return result


def _truncate_fp32_to_bf16_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _encode_e4m3_fn_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized FP8 E4M3 encoding using bit manipulation (round-to-nearest-even).

    Reference: torchao _f32_to_floatx_unpacked (OCP MX Formats).
    """
    p = _MX_DTYPE_PARAMS["fp8_e4m3"]
    shift = _F32_MBITS - p["mbits"]  # 23 - 3 = 20
    magic_adder = np.int32((1 << (shift - 1)) - 1)
    denorm_exp = (_F32_EXP_BIAS - p["exp_bias"]) + shift + 1  # 141
    denorm_mask_int = np.int32(denorm_exp << _F32_MBITS)
    denorm_mask_float = np.array(denorm_mask_int, dtype=np.int32).view(np.float32)
    max_code = np.uint8(0x7E)
    val_to_add = np.int32(((p["exp_bias"] - _F32_EXP_BIAS) << _F32_MBITS) + int(magic_adder))

    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.int32)
    sign = ((bits >> 24) & np.int32(0x80)).astype(np.uint8)
    abs_bits = bits & np.int32(0x7FFFFFFF)
    abs_val = abs_bits.view(np.float32).copy()

    nan_mask = np.isnan(values)
    saturate_mask = abs_val >= np.float32(p["max_pos"])
    denormal_mask = (~saturate_mask) & (abs_val < np.float32(p["min_normal"])) & (~nan_mask)
    normal_mask = (~saturate_mask) & (~denormal_mask) & (~nan_mask)

    # Denormal path
    denorm_result = (abs_val + denorm_mask_float).view(np.int32) - denorm_mask_int
    denorm_result = denorm_result.astype(np.uint8)

    # Normal path: adjust exponent and round-to-nearest-even
    mant_odd = ((abs_bits >> np.int32(shift)) & np.int32(1)).astype(np.int32)
    normal_result = abs_bits + val_to_add + mant_odd
    normal_result = ((normal_result >> np.int32(shift)) & np.int32(0x7F)).astype(np.uint8)

    # Combine branches
    result = np.where(saturate_mask, max_code, np.uint8(0))
    result = np.where(denormal_mask, denorm_result, result)
    result = np.where(normal_mask, normal_result, result)
    result = np.where(nan_mask, np.uint8(0x7F), result)
    result = result | sign
    return result.astype(np.uint8)


def _encode_e2m1_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized FP4 E2M1 encoding matching PTO MXFP4_E2M1 magic rounding."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.int32)
    sign = ((bits >> 28) & np.int32(0x8)).astype(np.uint8)
    abs_val = np.abs(values).astype(np.float32)

    nan_mask = np.isnan(values)
    inf_mask = np.isinf(abs_val)
    finite_mask = ~(nan_mask | inf_mask)

    result = np.zeros(values.shape, dtype=np.uint8)
    finite_abs = abs_val[finite_mask]
    if finite_abs.size:
        finite_bits = finite_abs.view(np.int32)
        biased_exp = ((finite_bits >> _F32_MBITS) & np.int32(0xFF)).astype(np.int32)
        biased_exp = np.clip(biased_exp, 127, 129).astype(np.int32)
        magic_bits = ((biased_exp + np.int32(22)) << np.int32(_F32_MBITS)).astype(np.int32)
        magic = magic_bits.view(np.float32)
        q = (finite_abs + magic).view(np.int32) - magic_bits
        base_code = (biased_exp - np.int32(127)) << np.int32(1)
        mag_code = np.minimum(q + base_code, np.int32(7)).astype(np.uint8)
        result[finite_mask] = mag_code

    result[inf_mask] = np.uint8(0x7)
    result[nan_mask] = np.uint8(0x7)
    result = result | np.where(nan_mask, np.uint8(0), sign)
    return result.astype(np.uint8)


def _pack_fp4_e2m1x2_low_first(codes: np.ndarray) -> np.ndarray:
    """Pack logical E2M1 nibbles as PTO MXFP4 output: even column in low nibble."""
    codes = np.asarray(codes, dtype=np.uint8)
    last_dim = codes.shape[-1]
    packed_shape = list(codes.shape)
    packed_shape[-1] = (last_dim + 1) // 2
    packed = np.zeros(packed_shape, dtype=np.uint8)
    packed[..., :(last_dim + 1) // 2] = codes[..., 0::2] & np.uint8(0x0F)
    if last_dim > 1:
        packed[..., :last_dim // 2] |= (codes[..., 1::2] & np.uint8(0x0F)) << np.uint8(4)
    return packed


def _pack_identity(codes: np.ndarray) -> np.ndarray:
    return codes


_MX_DTYPE_IMPLS = {
    "fp8_e4m3": {
        "encode": _encode_e4m3_fn_vectorized,
        "pack": _pack_identity,
    },
    "fp4_e2m1x2": {
        "encode": _encode_e2m1_vectorized,
        "pack": _pack_fp4_e2m1x2_low_first,
    },
}


def _float_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def _bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _quantmx_parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = parse_list_str(value)
    return [int(v) for v in value]


# Exponent ranges for values generated as mantissa in [1, 2) times 2^exp.
# Bounds include subnormal and one overflow exponent so tests can explicitly cover
# subnormal and Inf input generation through exp_range when requested.
_QUANTMX_EXP_RANGE = {
    "fp16": (-24, 16),
    "bf16": (-133, 128),
    "fp32": (-149, 128),
}

_QUANTMX_SUBNORMAL_EXP = {
    "fp16": -24,
    "bf16": -133,
    "fp32": -149,
}


def _quantmx_validate_dtype(dtype_name: str):
    assert dtype_name in _QUANTMX_EXP_RANGE, f"QuantMX golden does not support dtype {dtype_name}."


def _quantmx_resolve_exp_range(dtype_name: str, params: dict) -> list:
    _quantmx_validate_dtype(dtype_name)
    valid_lo, valid_hi = _QUANTMX_EXP_RANGE[dtype_name]
    exp_range = _quantmx_parse_int_list(params.get("exp_range"))
    if exp_range is not None:
        assert len(exp_range) == 2, "QuantMX exp_range must contain [min_exp, max_exp]."
        assert exp_range[0] <= exp_range[1], "QuantMX exp_range min must be <= max."
        assert valid_lo <= exp_range[0] <= valid_hi, (
            f"QuantMX exp_range min {exp_range[0]} is out of valid range [{valid_lo}, {valid_hi}] for {dtype_name}."
        )
        assert valid_lo <= exp_range[1] <= valid_hi, (
            f"QuantMX exp_range max {exp_range[1]} is out of valid range [{valid_lo}, {valid_hi}] for {dtype_name}."
        )
        return [int(exp_range[0]), int(exp_range[1])]

    default_ranges = {
        "fp32": [-40, 40],
        "bf16": [-80, 80],
        "fp16": [-20, 15],
    }
    default_range = default_ranges.get(dtype_name)
    assert default_range is not None, f"QuantMX golden does not support dtype {dtype_name}."
    default_lo, default_hi = default_range
    return [max(valid_lo, default_lo), min(valid_hi, default_hi)]


def _quantmx_has_exp_range_override(params: dict) -> bool:
    return str_to_bool(params.get("use_exp_range")) and params.get("exp_range") is not None


def _quantmx_resolve_data_range(input_tensor: dict) -> list:
    data_range = input_tensor.get("data_range")
    assert data_range is not None, "QuantMX golden requires input_datarange."
    range_lo = float(data_range["min"])
    range_hi = float(data_range["max"])
    assert np.isfinite(range_lo) and np.isfinite(range_hi), "QuantMX input_datarange must be finite."
    assert range_lo <= range_hi, "QuantMX input_datarange min must be <= max."
    return [range_lo, range_hi]


def _generate_quantmx_input_from_datarange(shape: tuple, data_range: list) -> np.ndarray:
    range_lo, range_hi = data_range
    if range_lo == range_hi:
        return np.full(shape, np.float32(range_lo), dtype=np.float32)
    return np.random.uniform(range_lo, range_hi, size=shape).astype(np.float32)


def _quantmx_special_exponents(dtype_name: str, exp_range: list) -> list:
    exp_lo, exp_hi = exp_range
    if exp_hi - exp_lo <= 32:
        return list(range(exp_lo, exp_hi + 1))

    default_values = {
        "fp32": [-120, -80, -32, -8, -1, 0, 1, 8, 32, 80, 120],
        "bf16": [-120, -80, -60, -16, -8, -1, 0, 1, 8, 16, 60, 80, 120],
        "fp16": [-24, -20, -14, -8, -1, 0, 1, 8, 12, 15],
    }
    values = default_values.get(dtype_name)
    assert values is not None, f"QuantMX golden does not support dtype {dtype_name}."
    candidates = values + [exp_lo, exp_hi, exp_lo + 1, exp_hi - 1, (exp_lo + exp_hi) // 2]
    clipped = sorted({min(exp_hi, max(exp_lo, item)) for item in candidates})
    return clipped


def _quantmx_inject_special_values(base: np.ndarray, dtype_name: str, exp_range: list):
    special_exponents = _quantmx_special_exponents(dtype_name, exp_range)
    if not special_exponents:
        return
    special_values = [np.float32(0.0), np.float32(-0.0)]
    for exp in special_exponents:
        special_values.extend(
            [
                np.float32(math.ldexp(1.0, exp)),
                np.float32(-math.ldexp(1.0, exp)),
                np.float32(math.ldexp(1.5, exp)),
                np.float32(-math.ldexp(1.25, exp)),
            ]
        )

    flat = base.reshape(-1)
    if flat.size == 0:
        return

    stride = max(1, flat.size // len(special_values))
    for idx, value in enumerate(special_values):
        pos = (idx * stride + idx * idx * 5) % flat.size
        flat[pos] = value


def _quantmx_special_value_count(size: int) -> int:
    if size <= 0:
        return 0
    return max(1, math.ceil(size / 10000))


def _quantmx_positions(size: int, count: int, offset: int, occupied: set) -> list:
    if size <= 0 or count <= 0:
        return []
    stride = max(1, size // count)
    positions = []
    for idx in range(count):
        pos = (offset + idx * stride + idx * idx * 17) % size
        probe = 0
        while pos in occupied and probe < size:
            pos = (pos + 1) % size
            probe += 1
        if pos in occupied:
            break
        occupied.add(pos)
        positions.append(pos)
    return positions


def _quantmx_inject_requested_values(casted: np.ndarray, dtype_name: str, params: dict):
    enable_subnormal = str_to_bool(params.get("enable_subnormal"))
    enable_inf = str_to_bool(params.get("enable_inf"))
    enable_nan = str_to_bool(params.get("enable_nan"))
    if not (enable_subnormal or enable_inf or enable_nan):
        return casted

    flat = casted.reshape(-1)
    count = _quantmx_special_value_count(flat.size)
    occupied = set()

    if enable_subnormal:
        positions = _quantmx_positions(flat.size, count, 0, occupied)
        subnormal = np.float32(math.ldexp(1.0, _QUANTMX_SUBNORMAL_EXP[dtype_name]))
        for idx, pos in enumerate(positions):
            flat[pos] = subnormal if idx % 2 == 0 else -subnormal

    if enable_inf:
        positions = _quantmx_positions(flat.size, count, flat.size // 3, occupied)
        for idx, pos in enumerate(positions):
            flat[pos] = np.inf if idx % 2 == 0 else -np.inf

    if enable_nan:
        positions = _quantmx_positions(flat.size, count, (flat.size * 2) // 3, occupied)
        for pos in positions:
            flat[pos] = np.nan

    return casted


def _generate_quantmx_input(input_tensor: dict, config: dict) -> np.ndarray:
    params = config.get("params", {}) or {}
    dtype_name = input_tensor["dtype"]
    shape = tuple(input_tensor["shape"])
    np_dtype = get_dtype_by_name(dtype_name)
    _quantmx_validate_dtype(dtype_name)

    explicit_exp_range = _quantmx_has_exp_range_override(params)
    data_range = _quantmx_resolve_data_range(input_tensor)

    if not explicit_exp_range:
        reshaped = _generate_quantmx_input_from_datarange(shape, data_range)
        casted = reshaped.astype(np_dtype)
        casted = _quantmx_inject_requested_values(casted, dtype_name, params)
        return casted

    exp_range = _quantmx_resolve_exp_range(dtype_name, params)
    exp_lo, exp_hi = exp_range

    exponents = np.random.randint(exp_lo, exp_hi + 1, size=shape)
    mantissas = np.random.uniform(1.0, 2.0, size=shape).astype(np.float32)
    signs = np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=shape)
    reshaped = np.ldexp(mantissas, exponents).astype(np.float32) * signs
    casted = reshaped.astype(np_dtype)
    casted = _quantmx_inject_requested_values(casted, dtype_name, params)
    return casted


@TestCaseLoader.reg_params_handler(ops=["QuantMX"])
def params_quantmx_func(params: dict):
    params["mode"] = params.get("mode") or "ROUND_DOWN"
    assert params["mode"] in ("ROUND_UP", "ROUND_DOWN"), "mode must be ROUND_UP or ROUND_DOWN"
    params["axis"] = int(params.get("axis", -1))
    params["performance_mode"] = str_to_bool(params.get("performance_mode"))
    params["use_exp_range"] = str_to_bool(params.get("use_exp_range"))
    params["exp_range"] = _quantmx_parse_int_list(params.get("exp_range")) if params["use_exp_range"] else None
    params["enable_subnormal"] = str_to_bool(params.get("enable_subnormal"))
    params["enable_inf"] = str_to_bool(params.get("enable_inf"))
    params["enable_nan"] = str_to_bool(params.get("enable_nan"))
    return params


def _quantmx_parse_golden_config(config: dict):
    params = config.get("params", {}) or {}
    mode = params.get("mode", "ROUND_DOWN")
    axis = int(params.get("axis", -1))
    output_dtype = config.get("output_tensors", [{}])[0].get("dtype", "fp8e4m3")
    if output_dtype not in _QUANTMX_OUTPUT_TO_DTYPE:
        raise ValueError(f"QuantMX golden does not support output dtype: {output_dtype}.")
    if mode not in _QUANTMX_SUPPORTED_MODES:
        raise ValueError(f"QuantMX golden does not support scale mode: {mode}.")
    quant_dtype = _QUANTMX_OUTPUT_TO_DTYPE[output_dtype]
    return mode, axis, quant_dtype, _MX_DTYPE_PARAMS[quant_dtype], _MX_DTYPE_IMPLS[quant_dtype]


def _quantmx_normalize_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"QuantMX axis is out of range. axis={axis}, rank={rank}")
    if axis != rank - 1 and not (rank >= 2 and axis == rank - 2):
        raise ValueError(f"QuantMX golden only supports axis=-1 and axis=-2. axis={axis}, rank={rank}")
    return axis


class _QuantMXGroupedInput(NamedTuple):
    rows: int
    cols: int
    group_cols: int
    scale_group_cols: int
    padded_cols: int
    x_grouped: np.ndarray


def _quantmx_group_input(x: np.ndarray, group_size: int):
    cols = x.shape[-1]
    rows = x.size // cols
    group_cols = (cols + group_size - 1) // group_size
    scale_group_cols = (cols + 63) // 64
    padded_cols = group_cols * group_size
    x_flat = x.reshape(rows, cols)
    x_padded = np.zeros((rows, padded_cols), dtype=np.float32)
    x_padded[:, :cols] = x_flat
    x_grouped = x_padded.reshape(rows, group_cols, group_size)
    return _QuantMXGroupedInput(rows, cols, group_cols, scale_group_cols, padded_cols, x_grouped)


def _quantmx_golden_max_source(
    x_grouped: np.ndarray, src_dtype_name: str, is_fp4_e2m1: bool, is_nv: bool, use_plain_fp8_max_abs: bool
):
    max_source = np.abs(x_grouped).astype(np.float32)
    if use_plain_fp8_max_abs:
        return max_source
    if is_nv and src_dtype_name == "fp16":
        return np.abs(x_grouped.astype(np.float16)).astype(np.float32)
    if is_nv and src_dtype_name == "bf16":
        return np.abs(x_grouped.astype(bfloat16)).astype(np.float32)
    if is_fp4_e2m1 and src_dtype_name == "fp16":
        return np.abs(_truncate_fp32_to_bf16_float32(x_grouped)).astype(np.float32)
    if is_fp4_e2m1 and src_dtype_name == "bf16":
        return np.abs(x_grouped.astype(bfloat16)).astype(np.float32)
    return max_source


def _quantmx_compute_exp_scaling(max_abs: np.ndarray, dp: dict, is_fp4_e2m1: bool, is_nv: bool):
    if is_nv:
        e8m0 = _compute_shared_exponents_nv(max_abs, dp["max_pos"])
        return e8m0, _compute_scalings_from_exponents_math(e8m0)
    if is_fp4_e2m1:
        e8m0 = _compute_shared_exponents_ocp_math(max_abs, dp["target_max_pow2"])
        return e8m0, _compute_scalings_from_exponents_math(e8m0)
    e8m0 = _compute_shared_exponents_floor(max_abs, dp["target_max_pow2"])
    return e8m0, _compute_scalings_from_exponents(e8m0)


def _quantmx_scale_grouped(
    x_grouped: np.ndarray, group_scaling: np.ndarray, src_dtype_name: str, is_fp4_e2m1: bool, is_nv: bool
):
    group_scaling_broadcast = group_scaling[:, :, np.newaxis]
    scaled = x_grouped * group_scaling_broadcast
    if is_fp4_e2m1 and src_dtype_name == "bf16":
        return (
            (x_grouped.astype(bfloat16).astype(np.float32) * group_scaling_broadcast.astype(np.float32))
            .astype(bfloat16)
            .astype(np.float32)
        )
    if is_nv and src_dtype_name == "fp16":
        return x_grouped * group_scaling_broadcast.astype(bfloat16).astype(np.float32)
    if is_nv and src_dtype_name == "bf16":
        return (
            (
                x_grouped.astype(bfloat16).astype(np.float32)
                * group_scaling_broadcast.astype(bfloat16).astype(np.float32)
            )
            .astype(bfloat16)
            .astype(np.float32)
        )
    return scaled


def _quantmx_build_exp(x_shape: tuple, rows: int, scale_group_cols: int, group_cols: int, e8m0: np.ndarray):
    exp_shape = list(x_shape[:-1]) + [scale_group_cols, 2]
    exp = np.zeros(exp_shape, dtype=np.uint8)
    exp_flat = exp.reshape(rows, scale_group_cols * 2)
    exp_flat[:, :group_cols] = e8m0.reshape(rows, group_cols)
    return exp


def _quantmx_axis_last_golden(
    x: np.ndarray,
    src_dtype_name: str,
    is_fp4_e2m1: bool,
    is_nv: bool,
    use_plain_fp8_max_abs: bool,
    dp: dict,
    impl: dict,
):
    group_size = 32
    grouped = _quantmx_group_input(x, group_size)
    max_source = _quantmx_golden_max_source(
        grouped.x_grouped, src_dtype_name, is_fp4_e2m1, is_nv, use_plain_fp8_max_abs
    )
    max_abs = np.max(max_source, axis=2).astype(np.float32)
    e8m0, group_scaling = _quantmx_compute_exp_scaling(max_abs, dp, is_fp4_e2m1, is_nv)
    scaled = _quantmx_scale_grouped(grouped.x_grouped, group_scaling, src_dtype_name, is_fp4_e2m1, is_nv)
    quant_grouped = impl["encode"](scaled)

    quant_flat = quant_grouped.reshape(grouped.rows, grouped.padded_cols)[:, :grouped.cols]
    quant = quant_flat.reshape(x.shape)
    quant = impl["pack"](quant)
    return [quant, _quantmx_build_exp(x.shape, grouped.rows, grouped.scale_group_cols, grouped.group_cols, e8m0)]


def _quantmx_axis_dn_golden(
    x: np.ndarray,
    src_dtype_name: str,
    is_fp4_e2m1: bool,
    is_nv: bool,
    use_plain_fp8_max_abs: bool,
    dp: dict,
    impl: dict,
):
    group_size = 32
    m_dim = x.shape[-2]
    n_dim = x.shape[-1]
    if m_dim % 64 != 0:
        raise ValueError(f"QuantMX axis=-2 golden requires second-last dim 64-aligned. Current dim: {m_dim}")

    prefix_shape = x.shape[:-2]
    prefix_rows = int(np.prod(prefix_shape)) if prefix_shape else 1
    group_rows = m_dim // group_size
    scale_rows = m_dim // 64
    x_3d = x.reshape(prefix_rows, m_dim, n_dim)
    x_grouped = x_3d.reshape(prefix_rows, group_rows, group_size, n_dim)
    x_grouped = x_grouped.transpose(0, 3, 1, 2).reshape(prefix_rows * n_dim, group_rows, group_size)

    max_source = _quantmx_golden_max_source(x_grouped, src_dtype_name, is_fp4_e2m1, is_nv, use_plain_fp8_max_abs)
    max_abs = np.max(max_source, axis=2).astype(np.float32)
    e8m0, group_scaling = _quantmx_compute_exp_scaling(max_abs, dp, is_fp4_e2m1, is_nv)
    scaled = _quantmx_scale_grouped(x_grouped, group_scaling, src_dtype_name, is_fp4_e2m1, is_nv)
    quant_grouped = impl["encode"](scaled)

    quant = quant_grouped.reshape(prefix_rows, n_dim, group_rows, group_size)
    quant = quant.transpose(0, 2, 3, 1).reshape(prefix_rows, m_dim, n_dim).reshape(x.shape)
    quant = impl["pack"](quant)

    exp_pairs = e8m0.reshape(prefix_rows, n_dim, group_rows).transpose(0, 2, 1)
    exp_pairs = exp_pairs.reshape(prefix_rows, scale_rows, 2, n_dim).transpose(0, 1, 3, 2)
    exp_public = exp_pairs.reshape(tuple(prefix_shape) + (scale_rows, n_dim, 2)).astype(np.uint8, copy=False)
    return [quant, exp_public]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestQuantMX/QuantMXOperationTest.TestQuantMX",
    ]
)
def gen_quantmx_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, _config: dict):
        mode, axis, quant_dtype, dp, impl = _quantmx_parse_golden_config(_config)
        input_tensor_desc = _config.get("input_tensors", [{}])[0]
        src_dtype_name = input_tensor_desc.get("dtype", "")
        is_fp4_e2m1 = quant_dtype == "fp4_e2m1x2"
        is_nv = mode == "ROUND_UP"
        use_plain_fp8_max_abs = (not is_fp4_e2m1) and (not is_nv)

        x = inputs[0].astype(np.float32, copy=False)
        if x.ndim < 1 or x.ndim > 4:
            raise ValueError("QuantMX golden only supports 1D to 4D input.")

        normalized_axis = _quantmx_normalize_axis(axis, x.ndim)
        if normalized_axis == x.ndim - 2:
            return _quantmx_axis_dn_golden(x, src_dtype_name, is_fp4_e2m1, is_nv, use_plain_fp8_max_abs, dp, impl)
        return _quantmx_axis_last_golden(x, src_dtype_name, is_fp4_e2m1, is_nv, use_plain_fp8_max_abs, dp, impl)

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("QuantMX", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseRightShift/BitwiseRightShiftOperationTest.TestBitwiseRightShift",
    ]
)
def gen_bitwise_right_shift_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        if not x0.is_signed():
            y = np.right_shift(x0.numpy(), x1.numpy())
        else:
            y = torch.bitwise_right_shift(x0, x1).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseRightShift", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseLeftShift/BitwiseLeftShiftOperationTest.TestBitwiseLeftShift",
    ]
)
def gen_bitwise_left_shift_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        if not x0.is_signed():
            y = np.left_shift(x0.numpy(), x1.numpy())
        else:
            y = torch.bitwise_left_shift(x0, x1).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseLeftShift", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseRightShifts/BitwiseRightShiftsOperationTest.TestBitwiseRightShifts",
    ]
)
def gen_bitwise_right_shifts_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x0 = torch.tensor(inputs[0])
        params["scalar_type"] = params.get("scalar_type", "int32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        if not x0.is_signed():
            y = np.right_shift(x0.numpy(), params["scalar"])
        else:
            y = torch.bitwise_right_shift(x0, params["scalar"]).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseRightShifts", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBitwiseLeftShifts/BitwiseLeftShiftsOperationTest.TestBitwiseLeftShifts",
    ]
)
def gen_bitwise_left_shifts_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x0 = torch.tensor(inputs[0])
        params["scalar_type"] = params.get("scalar_type", "int32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        if not x0.is_signed():
            y = np.left_shift(x0.numpy(), params["scalar"])
        else:
            y = torch.bitwise_left_shift(x0, params["scalar"]).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BitwiseLeftShifts", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSBitwiseRightShift/SBitwiseRightShiftOperationTest.TestSBitwiseRightShift",
    ]
)
def gen_s_bitwise_right_shift_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x1 = torch.tensor(inputs[0])
        params["scalar_type"] = params.get("scalar_type", "int32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        if not x1.is_signed():
            y = np.right_shift(params["scalar"], x1.numpy())
        else:
            y = torch.bitwise_right_shift(params["scalar"], x1).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("SBitwiseRightShift", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSBitwiseLeftShift/SBitwiseLeftShiftOperationTest.TestSBitwiseLeftShift",
    ]
)
def gen_s_bitwise_left_shift_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x1 = torch.tensor(inputs[0])
        params["scalar_type"] = params.get("scalar_type", "int32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        if not x1.is_signed():
            y = np.left_shift(params["scalar"], x1.numpy())
        else:
            y = torch.bitwise_left_shift(params["scalar"], x1).numpy()
        return [y]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("SBitwiseLeftShift", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCopySign/CopySignOperationTest.TestCopySign",
    ]
)
def gen_bitwise_right_shift_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:  # noqa: F811
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = from_numpy(inputs[0])
        x1 = from_numpy(inputs[1])
        y = torch.copysign(x0, x1)
        return [to_numpy(y)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("CopySign", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(case_names=["TestIsFinite/IsFiniteOperationTest.TestIsFinite"])
def gen_isfinite_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        result = torch.isfinite(from_numpy(inputs[0]))
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("IsFinite", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(case_names=["TestIsNan/IsNanOperationTest.TestIsNan"])
def gen_isnan_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        result = torch.isnan(from_numpy(inputs[0]))
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("IsNan", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCeilDiv/CeilDivOperationTest.TestCeilDiv",
    ]
)
def gen_ceil_div_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        result = torch.ceil(torch.div(from_numpy(inputs[0]), from_numpy(inputs[1])))
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("CeilDiv", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCeilDivs/CeilDivsOperationTest.TestCeilDivs",
    ]
)
def gen_ceil_divs_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "int32")
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        result = torch.ceil(torch.div(from_numpy(inputs[0]), scalar))
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("CeilDivs", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFloorDiv/FloorDivOperationTest.TestFloorDiv",
    ]
)
def gen_floor_div_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        tensor1 = from_numpy(inputs[1]).npu()
        result = torch.floor_divide(tensor0, tensor1)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("FloorDiv", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFloorDivs/FloorDivsOperationTest.TestFloorDivs",
    ]
)
def gen_floor_divs_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "int32")
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        tensor0 = from_numpy(inputs[0]).npu()
        tensor1 = from_numpy(scalar).npu()
        result = torch.floor_divide(tensor0, tensor1)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("FloorDivs", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAxpy/AxpyOperationTest.TestAxpy",
    ]
)
def gen_axpy_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        alpha = params.get("alpha", 1.0)
        input_dtype = inputs[0].dtype
        y = torch.from_numpy(inputs[0].astype(np.float32)).to(torch.float32)
        x = torch.from_numpy(inputs[1].astype(np.float32)).to(torch.float32)
        result_tensor = torch.add(y, x, alpha=alpha)
        if input_dtype == bfloat16:
            result = result_tensor.numpy().astype(bfloat16)
        else:
            result = result_tensor.numpy().astype(input_dtype)
        return [result]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Axpy", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Quantize"])
def quantize_params_func(params: dict):
    """
    Parameter handler for Quantize operation.
    Converts parameter types from JSON strings to appropriate Python types.
    """
    params["dtype"] = int(params.get("dtype", "3"))  # Default to DT_INT8
    params["axis"] = int(params.get("axis", "-1"))
    # Convert use_zero_points from string to bool
    use_zero_points_str = params.get("use_zero_points", "False")
    if isinstance(use_zero_points_str, bool):
        params["use_zero_points"] = use_zero_points_str
    else:
        params["use_zero_points"] = str_to_bool(use_zero_points_str)
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestQuantize/QuantizeOperationTest.TestQuantize",
    ]
)
def gen_quantize_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def quantize_golden_func(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        """
        Golden implementation for Quantize operation.

        Supports:
        - Symmetric quantization: q = round(x * scale)
        - Asymmetric quantization: q = round(x * scale) + zero_points
        """

        def ascend_tcvt_int8(x: torch.Tensor) -> torch.Tensor:
            """使用 torch.quantize 风格的实现"""

            # Step 1: FP32 -> S32 (round to nearest even)
            s32 = torch.round(x).to(torch.int32)  # -0.528 -> 0

            # Step 2 & 3: S32 -> FP16 -> INT8 (saturation)
            # 由于 S32 是整数，直接饱和到 INT8 范围即可
            return torch.clamp(s32, -128, 127).to(torch.int8)

        def ascend_tcvt_uint8(src_fp32: torch.Tensor) -> torch.Tensor:
            """
            三段式转换，最终输出 uint8
            """

            # Step 1: FP32 -> S32 (CAST_RINT)
            src_s32 = torch.round(src_fp32).to(torch.int32)
            # -0.528 -> 0

            # Step 2: S32 -> FP16 (CAST_RINT)
            src_f16 = src_s32.to(torch.float16)

            # Step 3: FP16 -> uint8 (CAST_RINT, Saturation ON)
            # uint8 范围: [0, 255]
            dst_float = torch.round(src_f16.to(torch.float32))
            dst_clamped = torch.clamp(dst_float, min=0, max=255)  # 关键：min=0
            dst = dst_clamped.to(torch.uint8)
            return dst

        params = config.get("params")
        input_tensor = from_numpy(inputs[0])
        scale = from_numpy(inputs[1])

        # Handle 1D input: reshape to [1, n] for processing
        is_1d_input = input_tensor.dim() == 1
        if is_1d_input:
            n = input_tensor.shape[0]
            input_tensor = input_tensor.reshape(1, n)

        # Get output dtype from config
        output_dtype = config.get("output_tensors")[0].get("dtype")
        axis = int(params.get("axis", "-1"))

        # Validate: 1D input only supports axis=-1
        if is_1d_input and axis == -2:
            raise ValueError("1D input only supports axis=-1, axis=-2 is not supported")

        # Convert to target dtype
        if output_dtype == "int8":
            if axis == -1:
                quantized = ascend_tcvt_int8(input_tensor * scale[..., None])
            elif axis == -2:  # axis = -2
                quantized = ascend_tcvt_int8(input_tensor * scale[..., None, :])
        elif output_dtype == "uint8":
            zero_points = from_numpy(inputs[2])
            if axis == -1:
                quantized = ascend_tcvt_uint8(input_tensor * scale[..., None] + zero_points[..., None])
            elif axis == -2:  # axis = -2
                quantized = ascend_tcvt_uint8(input_tensor * scale[..., None, :] + zero_points[..., None, :])
        else:
            raise ValueError(f"Unsupported output dtype for quantize: {output_dtype}")

        # If input was 1D, reshape output back to 1D
        if is_1d_input:
            quantized = quantized.reshape(n)

        return [to_numpy(quantized)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Quantize", quantize_golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Dequantize"])
def dequantize_params_func(params: dict):
    """
    Parameter handler for Dequantize operation.
    Converts parameter types from JSON strings to appropriate Python types.
    """
    params["otype"] = int(params.get("otype", "0"))  # Default to DT_FP32
    params["axis"] = int(params.get("axis", "-1"))
    # Convert use_zero_points from string to bool
    use_zero_points_str = params.get("use_zero_points", "False")
    if isinstance(use_zero_points_str, bool):
        params["use_zero_points"] = use_zero_points_str
    else:
        params["use_zero_points"] = str_to_bool(use_zero_points_str)
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDequantize/DequantizeOperationTest.TestDequantize",
    ]
)
def gen_dequantize_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def dequantize_golden_func(
        inputs: List[np.ndarray],
        config: Dict[str, Any],  # noqa
    ) -> List[np.ndarray]:
        """
        Golden implementation for Dequantize operation.

        Supports:
        - INT8 -> FP32: symmetric/asymmetric
        - INT16 -> FP32: symmetric/asymmetric

        Formula:
        - Symmetric: dst = src * scale
        - Asymmetric: dst = (src - offset) * scale
        """
        params = config.get("params")
        input_tensor = inputs[0]  # INT8 or INT16
        scale = inputs[1]  # FP32

        axis = int(params.get("axis", "-1"))
        use_zero_points = params.get("use_zero_points", False)

        # Handle 1D input: reshape to [1, n] for processing
        is_1d_input = input_tensor.ndim == 1
        if is_1d_input:
            n = input_tensor.shape[0]
            input_tensor = input_tensor.reshape(1, n)

        # Validate: 1D input only supports axis=-1
        if is_1d_input and axis == -2:
            raise ValueError("1D input only supports axis=-1, axis=-2 is not supported")

        # Convert input to float for computation
        if input_tensor.dtype == np.int8:
            input_float = input_tensor.astype(np.float32)
        elif input_tensor.dtype == np.int16:
            input_float = input_tensor.astype(np.float32)
        else:
            input_float = input_tensor.astype(np.float32)

        # Broadcast scale based on axis
        ndim = input_float.ndim
        normalized_axis = axis if axis < 0 else axis - ndim

        if use_zero_points and len(inputs) > 2:
            zero_points = inputs[2].astype(np.float32)
            # Broadcast scale and zero_points based on axis
            if normalized_axis == -1:
                # Broadcast along last dimension
                if scale.ndim == ndim - 1:
                    scale = np.expand_dims(scale, axis=-1)
                if zero_points.ndim == ndim - 1:
                    zero_points = np.expand_dims(zero_points, axis=-1)
            elif normalized_axis == -2:
                # Broadcast along second-to-last dimension
                if scale.ndim == ndim - 1:
                    scale = np.expand_dims(scale, axis=-2)
                if zero_points.ndim == ndim - 1:
                    zero_points = np.expand_dims(zero_points, axis=-2)
            result = (input_float - zero_points) * scale
        else:
            # Broadcast scale based on axis
            if normalized_axis == -1:
                # Broadcast along last dimension
                if scale.ndim == ndim - 1:
                    scale = np.expand_dims(scale, axis=-1)
            elif normalized_axis == -2:
                # Broadcast along second-to-last dimension
                if scale.ndim == ndim - 1:
                    scale = np.expand_dims(scale, axis=-2)
            result = input_float * scale

        # If input was 1D, reshape output back to 1D
        if is_1d_input:
            result = result.reshape(n)

        # Output is always FP32
        return [result.astype(np.float32)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Dequantize", dequantize_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSinh/SinhOperationTest.TestSinh",
    ]
)
def gen_sinh_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.sinh(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Sinh", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCosh/CoshOperationTest.TestCosh",
    ]
)
def gen_cosh_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.cosh(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Cosh", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAsin/AsinOperationTest.TestAsin",
    ]
)
def gen_asin_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.asin(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Asin", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAcos/AcosOperationTest.TestAcos",
    ]
)
def gen_acos_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.acos(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Acos", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestASinh/ASinhOperationTest.TestASinh",
    ]
)
def gen_asinh_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.asinh(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("ASinh", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestACosh/ACoshOperationTest.TestACosh",
    ]
)
def gen_acosh_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.acosh(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("ACosh", generate_wrapper, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAtanh/AtanhOperationTest.TestAtanh",
    ]
)
def gen_atanh_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def generate_wrapper(
        inputs: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        tensor0 = from_numpy(inputs[0]).npu()
        result = torch.atanh(tensor0)
        result = result.cpu()
        return [to_numpy(result)]

    logging.debug(f"Generating golden files of {case_name} ...")
    return gen_op_golden("Atanh", generate_wrapper, output, case_index)


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestAdd/AddOperationTest.TestAdd",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_add_op_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
