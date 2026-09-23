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

import math

import ml_dtypes
import numpy as np
import torch


def is_number(input_str: str):
    try:
        num = float(input_str)
        # not support parse inf nan in c++ json parser
        return not math.isinf(num) and not math.isnan(num)
    except ValueError:
        return False


def parse_number(input_str: str):
    try:
        return int(input_str)
    except ValueError:
        return float(input_str)


def parse_list_str(input_str: str):
    if input_str is None:
        raise ValueError("Can't convert None to list.")
    input_str = input_str.replace(" ", "")
    if input_str.startswith("[") and input_str.endswith("]"):
        input_str = input_str[1:-1]

    ret_list = []
    element_split_ident = " "
    if "{" in input_str:
        element_split_ident = "},{"
    if "[" in input_str:
        element_split_ident = "],["
    if element_split_ident in input_str:
        for sub_str in input_str.split(element_split_ident):
            ret_list.append(parse_list_str(sub_str))
    else:
        for sub_str in input_str.split(","):
            if not is_number(sub_str):
                ret_list.append(sub_str)
            else:
                ret_list.append(parse_number(sub_str))
    return ret_list


def str_to_bool(input_str: str):
    if input_str is None:
        return False
    input_str = str(input_str).strip().upper()
    return input_str in ("TRUE", "1")


def get_dtype_by_name(name: str, is_torch: bool = False, check: bool = True):
    # torch FP8 dtype
    fp8_e4m3_torch = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None
    fp8_e5m2_torch = torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else None
    fp8_e8m0_torch = torch.float8_e8m0fnu if hasattr(torch, 'float8_e8m0fnu') else None

    str_to_dtype = {
        "int8": [np.int8, torch.int8],
        "int16": [np.int16, torch.int16],
        "int32": [np.int32, torch.int32],
        "int64": [np.int64, torch.int64],
        "fp16": [np.float16, torch.float16],
        "fp32": [np.float32, torch.float32],
        "fp64": [np.float64, torch.float64],
        "uint8": [np.uint8, torch.uint8],
        "uint16": [np.uint16, torch.uint16 if hasattr(torch, 'uint16') else None],
        "uint32": [np.uint32, torch.uint32 if hasattr(torch, 'uint32') else None],
        "uint64": [np.uint64, torch.uint64 if hasattr(torch, 'uint64') else None],
        "bool": [np.bool_, torch.bool],
        "double": [np.float64, torch.double],
        "complex64": [np.complex64, torch.complex64],
        "complex128": [np.complex128, torch.complex64],
        "bf16": [ml_dtypes.bfloat16, torch.bfloat16],
        "fp4_e2m1x2": [np.uint8, torch.uint8],
        "fp4_e1m2x2": [np.uint8, torch.uint8],
        "fp8e4m3": [
            np.uint8,
            fp8_e4m3_torch if fp8_e4m3_torch is not None else np.uint8,
        ],
        "fp8e5m2": [
            np.uint8,
            fp8_e5m2_torch if fp8_e5m2_torch is not None else np.uint8,
        ],
        "fp8e8m0": [np.uint8, fp8_e8m0_torch if fp8_e8m0_torch is not None else np.uint8],
        "hf8": [np.uint8, torch.uint8],
    }
    return str_to_dtype.get(name, [np.float32, torch.float32])[is_torch]


def parse_dict_str(input_str: str):
    if input_str is None:
        raise ValueError("Can't convert None to list.")
    input_str = input_str.replace(" ", "")
    if input_str.startswith("{") and input_str.endswith("}"):
        input_str = input_str[1:-1]

    key_values = input_str.split(",")
    res = {}
    value_index = 0
    while value_index < len(key_values):
        if ":" in key_values[value_index]:
            key, value = key_values[value_index].split(":")
            while value_index + 1 < len(key_values) and ":" not in key_values[value_index + 1]:
                value += "," + key_values[value_index + 1]
                value_index += 1
            res[key] = value
        value_index += 1
    return res
