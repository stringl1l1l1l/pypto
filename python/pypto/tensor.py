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
from typing import List, Optional, Sequence, Tuple, Union

import pypto

from . import pypto_impl
from ._element import Element
from ._utils import source_location, to_sym, to_syms
from .enum import (
    CachePolicy,
    DataType,
    DequantScaleRoundingMode,
    PrecisionType,
    TileOpFormat,
)
from .error import FeError
from .symbolic_scalar import SymbolicScalar, SymInt

LogicalTensor = pypto_impl.LogicalTensor


class TensorAnnotation:
    """Type annotation for Tensor parameters.

    This class is used to represent tensor type annotations in function signatures,
    supporting the Python-standard syntax: pypto.Tensor[shape, dtype]
    """

    def __init__(
        self,
        shape=None,
        dtype=None,
        name="",
        format=TileOpFormat.TILEOP_ND,
        data_ptr=None,
        device=None,
        ori_shape=None,
    ):
        self.shape = shape if shape is not None else []
        self.dtype = dtype if dtype is not None else pypto.DT_FP32
        self.name = name
        self.format = format
        self.data_ptr = data_ptr
        self.device = device
        self.ori_shape = ori_shape

    def to_tensor(self, name: str = ""):
        final_name = name if name else self.name
        return Tensor(self.shape, self.dtype, final_name, self.format, self.data_ptr, self.device, self.ori_shape)


class Tensor:
    def __init__(
        self,
        shape=None,
        dtype: Union[DataType, None] = None,
        name: str = "",
        format=None,
        data_ptr: Optional[int] = None,
        device=None,
        ori_shape=None,
    ):
        self.ori_shape = ori_shape
        self.status_shape = None
        # Mark explicit dtype and format configuration
        self.explicit_dtype = dtype
        self.explicit_format = format
        # Use default value if format is not explicitly passed
        ndtype = dtype or pypto.DT_FP32
        nformat = format or TileOpFormat.TILEOP_ND

        # Normalize shape
        if shape is None:
            nshape = []
        elif shape and all(isinstance(s, int) for s in shape):
            nshape = list(shape)
        elif isinstance(shape, list) and self._validate_status_shape(shape):
            nshape = []
            self.status_shape = shape
        else:
            nshape = to_syms(shape)
            if not isinstance(nshape, list):
                raise FeError(TypeError("shape must be a list of int or SymbolicScalar"))
        self._base = pypto_impl.Tensor(ndtype, nshape, name, nformat)
        self.data_ptr = data_ptr
        self.device = device

    @classmethod
    def __class_getitem__(cls, params):
        """Enable Tensor[shape, dtype] syntax for type annotations.

        Examples:
            >>> a: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32]
            >>> b: pypto.Tensor[[pypto.STATIC, ...], pypto.DT_INT32]
            >>> c: pypto.Tensor[[], pypto.DT_INT32]
            >>> d: pypto.Tensor[[batch_size, hidden_size], pypto.DT_FP32, pypto.TileOpFormat.TILEOP_ND]

            Note: Tensor[] (empty parameters) is NOT supported.

        Parameters
        ----------
        params : tuple
            A tuple of (shape, dtype).

        Returns
        -------
        TensorAnnotation
            A tensor annotation object for type hints.
        """
        if not isinstance(params, tuple):
            params = (params,)

        shape = None
        dtype = None
        name = ""
        format = None
        data_ptr = None
        device = None
        ori_shape = None

        for param in params:
            if isinstance(param, dict):
                name = param.get('name', name)
                format = param.get('format', format)
                data_ptr = param.get('data_ptr', data_ptr)
                device = param.get('device', device)
                ori_shape = param.get('ori_shape', ori_shape)
                if 'dtype' in param:
                    dtype = param['dtype']
                if 'shape' in param:
                    shape = param['shape']
            elif isinstance(param, (list, tuple)):
                shape = param
            elif isinstance(param, DataType):
                dtype = param
            elif isinstance(param, TileOpFormat):
                format = param
            elif isinstance(param, str):
                name = param
            else:
                pass

        return TensorAnnotation(shape, dtype, name, format, data_ptr, device, ori_shape)

    @source_location
    def __setitem__(self, key, value):
        """
        Set tensor data by index or slice.

        Args:
            key (Union[int, SymbolicScalar, slice]): Index or slice to set.
            value (Tensor | Element): value to set.

        example:
        # All slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2, 2), pypto.DT_FP32)
        a[0:, 0:] = b # assemble(b, (0, 0), a)
        Input a:[[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],]
        Input b:[[10, 10]
                 [10, 10]]
        Output a:[[10, 10, 0, 0],
                  [10, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]

        # Index and slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2), pypto.DT_FP32)
        a[0, 1:3] = b # reshape b to (1, 2), assemble(b, (0, 1), a)
        Input a:[[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],]
        Input b:[10, 10]
        Output a:[[0, 10, 10, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]

        # Negative index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2), pypto.DT_FP32)
        a[-1, -3:-1] = b # equivalent to a[3, 1:3]

        # Ellipsis index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2, 2), pypto.DT_FP32)
        a[..., 1:3] = b # equivalent to a[0:2, 1:3]

        # single data
        a = pypto.tensor((4, 4), pypto.DT_INT32)
        a[0, 0] = 1 #SetTensorData, supports only DT_INT32 tensors

        """

        if self._is_empty_slice(key):
            self.move(value)
            return

        if isinstance(key, slice) and isinstance(key.stop, Tensor):
            if not isinstance(key.start, int):
                raise FeError(TypeError(f"scatter key.start must be int, got {type(key.start).__name__}"))
            return pypto.scatter(self, key.start, key.stop, value)

        key = self._normalize_key(key)

        if all(isinstance(k, (int, SymbolicScalar)) for k in key):
            pypto_impl.SetTensorData(to_sym(value), to_syms(key), self._base)
            return

        if all(isinstance(k, slice) for k in key):
            offsets = self._get_assemble_offset(key, self.shape)
            return pypto.assemble(value, offsets, self)

        if all(isinstance(k, (slice, int, SymbolicScalar)) for k in key):
            new_shape = self._add_one_dim(key, value.shape)
            value_reshaped = pypto.reshape(value, new_shape)
            new_key, _ = self._get_slice_index(key)
            offsets = self._get_assemble_offset(tuple(new_key), self.shape)
            return pypto.assemble(value_reshaped, offsets, self)

        raise FeError(ValueError("tuple key must be int, SymbolicScalar or slice"))

    def __iter__(self):
        raise FeError(TypeError("Tensor is not iterable."))

    @source_location
    def __getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None):
        """
        Get tensor data by index, supporting integer indices, slices, ellipsis,
        and their combinations to retrieve sub-tensors.

        Args:
            key (Union[int, SymbolicScalar, slice, ellipsis]): Index or slice to get.

        Returns:
            Tensor | Element: tensor data.

        example:
        # All slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = a[:2, :2] # view(a, [2, 2], [0, 0])
        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:[[1, 2],
                  [5, 6]]

        # Index and slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = a[1, 1:3] # view(a, [1, 2], [1, 1]), then reshape to [2]

        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:[6, 7]

        # Negative index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = a[-1, -3:-1]# equivalent to a[3, 1:3]
        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:[14, 15]

        # Ellipsis index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = a[..., 1:3]# equivalent to a[0:4, 1:3]
        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:[[2, 3],
                  [6, 7],
                  [10, 11],
                  [14, 15]]

        # Less dim index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = a[1]# equivalent to a[1, :]
        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:[5, 6, 7, 8]

        # single data
        a = pypto.tensor((4, 4), pypto.DT_INT32)
        b = a[0, 0] #GetTensorData, supports only DT_INT32 tensors
        Input a:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output b:1
        """
        if self._is_empty_slice(key):
            return self

        if isinstance(key, slice) and isinstance(key.stop, Tensor):
            if not isinstance(key.start, int):
                raise FeError(TypeError(f"gather key.start must be int, got {type(key.start).__name__}"))
            return pypto.gather(self, key.start, key.stop)

        key = self._normalize_key(key)

        if all(isinstance(k, (int, SymbolicScalar)) for k in key):
            return pypto_impl.GetTensorData(self._base, to_syms(key))

        if all(isinstance(k, slice) for k in key):
            offsets, shapes = self._get_view_offset_shape(key, self.shape)
            return pypto.view(self, shapes, offsets, valid_shape=valid_shape)

        if all(isinstance(k, (slice, int, SymbolicScalar)) for k in key):
            new_key, bool_shape = self._get_slice_index(key)
            offsets, shapes = self._get_view_offset_shape(tuple(new_key), self.shape)
            res = pypto.view(self, shapes, offsets, valid_shape=valid_shape)
            res_shape = [res.shape[d] for d in range(res.dim) if bool_shape[d]]
            return pypto.reshape(res, res_shape)

        raise FeError(ValueError("tuple key must be int, SymbolicScalar or slice"))

    @source_location
    def __add__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    @source_location
    def __radd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    @source_location
    def __iadd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    @source_location
    def __sub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    @source_location
    def __isub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    @source_location
    def __mul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    @source_location
    def __imul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    @source_location
    def __truediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    @source_location
    def __itruediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    @source_location
    def __gt__(self, other: 'Tensor') -> 'Tensor':
        return self.greater(other)

    @source_location
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if other.dtype in {pypto.DT_FP16, pypto.DT_BF16, pypto.DT_FP32}:
            out_dtype = other.dtype
        elif other.dtype == pypto.DT_INT8:
            out_dtype = pypto.DT_INT32
        else:
            raise FeError(RuntimeError("unsupported dtype"))
        return pypto.matmul(self, other, out_dtype)

    @source_location
    def __neg__(self) -> 'Tensor':
        return pypto.neg(self)

    @source_location
    def __pos__(self) -> 'Tensor':
        return self

    @source_location
    def __rsub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.neg(self.sub(other))

    @source_location
    def __rmul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.mul(self, other)

    @source_location
    def __rtruediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.mul(pypto.reciprocal(self), other)

    @source_location
    def __floordiv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.floor(pypto.div(self, other))

    @source_location
    def __mod__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.remainder(self, other)

    @source_location
    def __pow__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.pow(self, other)

    @source_location
    def __lt__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.lt(self, other)

    @source_location
    def __le__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.le(self, other)

    @source_location
    def __ge__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.ge(self, other)

    @source_location
    def __eq__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.eq(self, other)

    @source_location
    def __ne__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.ne(self, other)

    @property
    def dtype(self) -> DataType:
        return self._base.GetDataType()

    @property
    def shape(self) -> List[SymInt]:
        if getattr(self, "status_shape", None) is not None:
            return self.status_shape

        out = []
        if self._base.IsEmpty():
            return out
        for i, n in enumerate(self._base.GetShape()):
            if n == -1:
                out.append(pypto_impl.GetInputShape(self._base, i))
            else:
                out.append(n)
        return out

    @property
    def valid_shape(self) -> List[SymInt]:
        """
        Return the valid shape of the tensor, debug purpose only.
        """
        return self._base.GetValidShape()

    @property
    def dim(self) -> int:
        return self._base.Dim()

    @property
    def id(self) -> int:
        return self._base.Id()

    @property
    def format(self) -> TileOpFormat:
        return self._base.Format()

    @property
    def name(self) -> str:
        return self._base.GetName()

    @name.setter
    def name(self, value: str) -> None:
        self._base.SetName(value)

    @staticmethod
    def _validate_status_shape(lst):
        """
        Validates the types of elements in the list:
        - All elements except the last one: only pypto.StatusType enum or int are allowed
        - The last element: pypto.StatusType, int or Ellipsis (...) are allowed
        :param lst: The list to be validated
        :return: (bool, str) → (validation result, validation message)
        """

        # boundary 1: empty list
        if not isinstance(lst, list):
            return False
        if len(lst) == 0:
            return True

        for idx, elem in enumerate(lst):
            # Check if this is the last element
            is_last = idx == len(lst) - 1

            # Case1: last element -> allows enum/int/...
            if is_last:
                # Check if it's an allowed type
                if (
                    isinstance(elem, pypto.StatusType)  # pypto.enum type
                    or isinstance(elem, int)  # int type
                    or elem is ...
                ):  # ellipsis
                    continue
                else:
                    return False

            # Case2: non-last element -> only allows enum/int
            else:
                if isinstance(elem, pypto.StatusType) or isinstance(elem, int):
                    continue
                else:
                    return False

        # All elements passed validation
        return True

    @staticmethod
    def _get_assemble_offset(key, shape):
        offsets = []
        for k in key:
            start, step = k.start, k.step
            if step not in (1, None):
                raise FeError(ValueError("step must be 1 or None"))
            if start is None:
                offsets.append(0)
            elif isinstance(start, (int, SymbolicScalar)):
                offsets.append(start)
        return offsets

    @staticmethod
    def _add_one_dim(key, value_shape):
        slices_count = sum(1 for k in key if isinstance(k, slice))
        if slices_count != len(value_shape):
            raise FeError(
                ValueError(
                    f"The number of slice in key ({slices_count}) "
                    f"must match the length of input Tensor ({len(value_shape)}). "
                )
            )
        new_shape = []
        idx = 0
        for k in key:
            if isinstance(k, slice):
                new_shape.append(value_shape[idx])
                idx += 1
            else:
                new_shape.append(1)
        return new_shape

    @staticmethod
    def _negative_index_to_positive(key, shape):
        normalized = []
        for axis, k in enumerate(key):
            size = shape[axis]
            if isinstance(k, (int, SymbolicScalar)):
                if isinstance(k, int) and k < 0:
                    k = size + k
                normalized.append(k)
                continue
            start, stop, step = k.start, k.stop, k.step
            if (
                isinstance(size, int)
                and size >= 0
                and (step is None or (isinstance(step, int) and step == 1))
                and all(value is None or isinstance(value, int) for value in (start, stop))
            ):
                start, stop, _ = slice(start, stop, step).indices(size)
            else:
                if isinstance(start, int) and start < 0:
                    start = size + start
                if isinstance(stop, int) and stop < 0:
                    stop = size + stop
            normalized.append(slice(start, stop, step))
        return tuple(normalized)

    @staticmethod
    def _get_slice_index(key):
        new_key = []
        bool_shape = []
        for k in key:
            if isinstance(k, (int, SymbolicScalar)):
                new_key.append(slice(k, k + 1))
                bool_shape.append(False)
            else:
                new_key.append(k)
                bool_shape.append(True)
        return new_key, bool_shape

    @staticmethod
    def _get_view_offset_shape(key, shape):
        offsets = []
        shapes = []
        for axis, k in enumerate(key):
            start, stop, step = k.start, k.stop, k.step
            if step != 1 and step is not None:
                raise FeError(ValueError("step must be 1 or None"))
            if start is None:
                start = 0
            if stop is None:
                stop = shape[axis]
            offsets.append(start)
            tshape = stop - start
            if isinstance(tshape, SymbolicScalar):
                tshape = int(tshape.simplify())
            shapes.append(tshape)  # shape should be concrete
        return offsets, shapes

    @classmethod
    def from_base(cls, base: pypto_impl.Tensor) -> 'Tensor':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    @classmethod
    def from_logical_tensor(cls, logical_tensor: 'LogicalTensor') -> 'Tensor':
        return cls.from_base(pypto_impl.Tensor(logical_tensor))

    def to_tensor(self, name: str = ""):
        """Return self for unified interface with TensorAnnotation."""
        if name:
            self.name = name
        return self

    def is_empty(self) -> bool:
        return self._base.IsEmpty()

    def copy(self, name: str = "") -> 'Tensor':
        name = name or self.name
        return Tensor(self._base.GetShape(), self.dtype, name, self.format)

    def set_cache_policy(self, policy: CachePolicy, value: bool) -> None:
        self._base.SetCachePolicy(policy, value)

    def get_cache_policy(self, policy: CachePolicy) -> bool:
        return self._base.GetCachePolicy(policy)

    def move(self, other: 'Tensor') -> None:
        if isinstance(other, Tensor):
            self._base.Move(other._base)
        else:
            raise FeError(TypeError(f"'{type(other).__name__}' type cannot be moved to Tensor"))

    def base(self) -> pypto_impl.Tensor:
        return self._base

    def logical_tensor(self) -> 'LogicalTensor':
        return self._base.LogicalTensor()

    def set_logical_tensor(self, logical_tensor: 'LogicalTensor') -> None:
        # Move/AssignStorage uniquifies IR names. Skip when storage is unchanged
        # so if-branch rollback does not rename function parameters.
        if not self.is_empty() and self.logical_tensor() is logical_tensor:
            return
        self._base.Move(pypto_impl.Tensor(logical_tensor))

    @source_location
    def add(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.add(self, other)

    @source_location
    def axpy_(self, x: 'Tensor', alpha: 'int | float' = 1.0) -> 'Tensor':
        return pypto.axpy_(self, x, alpha)

    @source_location
    def sub(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.sub(self, other)

    @source_location
    def mul(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.mul(self, other)

    @source_location
    def hypot(self, other: 'Tensor') -> 'Tensor':
        return pypto.hypot(self, other)

    @source_location
    def prelu(self, weight: 'Tensor') -> 'Tensor':
        return pypto.prelu(self, weight)

    @source_location
    def div(
        self,
        other: 'Tensor | int | float',
        precision_type: PrecisionType = PrecisionType.HIGH_PRECISION,
    ) -> 'Tensor':
        return pypto.div(self, other, precision_type)

    @source_location
    def fmod(
        self,
        other: 'Tensor | int | float',
        precision_type: PrecisionType = PrecisionType.HIGH_PRECISION,
    ) -> 'Tensor':
        return pypto.fmod(self, other, precision_type)

    @source_location
    def remainder(
        self,
        other: 'Tensor | int | float',
        precision_type: PrecisionType = PrecisionType.HIGH_PRECISION,
    ) -> 'Tensor':
        return pypto.remainder(self, other, precision_type)

    @source_location
    def greater(self, other: 'Tensor'):
        return pypto.greater(self, other)

    @source_location
    def fill_(self, other: 'int | float') -> 'Tensor':
        self.move(pypto.full(self.shape, other, self.dtype))
        return self

    @source_location
    def matmul(
        self, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=None
    ) -> "Tensor":
        return pypto.matmul(
            self,
            mat2,
            out_dtype,
            a_trans=a_trans,
            b_trans=b_trans,
            c_matrix_nz=c_matrix_nz,
            extend_params=extend_params,
        )

    @source_location
    def assemble(self, input: 'Tensor', offsets: List[Union[int, SymbolicScalar]]) -> None:
        """
        Assemble a small Tensor into a larger Tensor based on specified offsets.

        Args:
            input (Tensor): The small input tensor to be assembled into the larger tensor.
            offsets (Union[List[int], List[SymbolicScalar]]): Offset for placing the input tensor.

        example:
        s = pypto.tensor((16, 16), pypto.DT_FP32)
        a = pypto.tensor((2, 2), pypto.DT_FP32)
        s.assemble(a, [0, 0])
        """
        pypto.assemble(input, offsets, self)

    @source_location
    def reshape(
        self, shape: List[int], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False
    ) -> 'Tensor':
        if inplace:
            return pypto.reshape(self, shape, inplace=inplace)
        else:
            return pypto.reshape(self, shape, valid_shape=valid_shape, inplace=inplace)

    @source_location
    def unsqueeze(self, dim: int) -> 'Tensor':
        return pypto.unsqueeze(self, dim)

    @source_location
    def view(
        self,
        shape: List[int],
        offsets: List[Union[int, SymbolicScalar]],
        *,
        valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
    ) -> 'Tensor':
        return pypto.view(self, shape, offsets, valid_shape=valid_shape)

    @source_location
    def clone(self) -> 'Tensor':
        return pypto.clone(self)

    @source_location
    def sin(self) -> 'Tensor':
        return pypto.sin(self)

    @source_location
    def cos(self) -> 'Tensor':
        return pypto.cos(self)

    @source_location
    def sinh(self) -> 'Tensor':
        return pypto.sinh(self)

    @source_location
    def cosh(self) -> 'Tensor':
        return pypto.cosh(self)

    @source_location
    def asinh(self) -> 'Tensor':
        return pypto.asinh(self)

    @source_location
    def acosh(self) -> 'Tensor':
        return pypto.acosh(self)

    @source_location
    def sigmoid(self) -> 'Tensor':
        return pypto.sigmoid(self)

    @source_location
    def softmax(self, dim: int) -> 'Tensor':
        return pypto.softmax(self, dim)

    @source_location
    def maximum(self, other: 'Tensor') -> 'Tensor':
        return pypto.maximum(self, other)

    @source_location
    def where(
        self, condition: 'Tensor', x: Union['Tensor', Element, float], y: Union['Tensor', Element, float]
    ) -> 'Tensor':
        return pypto.where(self, condition, x, y)

    @source_location
    def lrelu(self, other: 'Tensor', negative_slope: Union[float, Element] = 0.01) -> 'Tensor':
        return pypto.lrelu(self, other, negative_slope)

    @source_location
    def topk(self, k: int, dim: Optional[int] = None, largest: bool = True) -> Tuple['Tensor', 'Tensor']:
        return pypto.topk(self, k, dim, largest)

    @source_location
    def quant_mx(
        self,
        quant_dtype: DataType = DataType.DT_FP8E4M3,
        mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
        axis: int = -1,
        performance_mode: bool = True,
    ) -> Tuple['Tensor', 'Tensor']:
        return pypto.quant_mx(self, quant_dtype, mode, axis, performance_mode)

    @source_location
    def sort32(self, index: Optional[int] = None) -> 'Tensor':
        return pypto.sort32(self, index)

    @source_location
    def mrgsort(self, mergesize: int) -> 'Tensor':
        return pypto.mrgsort(self, mergesize)

    @source_location
    def exp(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.exp(self, precision_type)

    @source_location
    def sign(self) -> 'Tensor':
        return pypto.sign(self)

    @source_location
    def signbit(self) -> 'Tensor':
        return pypto.signbit(self)

    def tanh(self) -> 'Tensor':
        return pypto.tanh(self)

    @source_location
    def exp2(self) -> 'Tensor':
        return pypto.exp2(self)

    @source_location
    def expm1(self) -> 'Tensor':
        return pypto.expm1(self)

    @source_location
    def erfc(self) -> 'Tensor':
        return pypto.erfc(self)

    @source_location
    def log(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.log(self, precision_type)

    @source_location
    def log1p(self) -> 'Tensor':
        return pypto.log1p(self)

    @source_location
    def log10(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.log10(self, precision_type)

    @source_location
    def log2(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.log2(self, precision_type)

    @source_location
    def logical_not(self) -> 'Tensor':
        return pypto.logical_not(self)

    @source_location
    def amax(self, dim: int, keepdim: bool = False) -> 'Tensor':
        return pypto.amax(self, dim, keepdim)

    @source_location
    def amin(self, dim: int, keepdim: bool = False) -> 'Tensor':
        return pypto.amin(self, dim, keepdim)

    @source_location
    def argmax(self, dim: int, keepdim: bool = False) -> 'Tensor':
        return pypto.argmax(self, dim, keepdim)

    @source_location
    def argmin(self, dim: int, keepdim: bool = False) -> 'Tensor':
        return pypto.argmin(self, dim, keepdim)

    @source_location
    def sum(self, dim: int, keepdim: bool = False) -> 'Tensor':
        return pypto.sum(self, dim, keepdim)

    @source_location
    def round(self, decimals: int = 0) -> 'Tensor':
        return pypto.round(self, decimals)

    @source_location
    def rsqrt(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.rsqrt(self, precision_type)

    @source_location
    def sqrt(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor':
        return pypto.sqrt(self, precision_type)

    @source_location
    def ceil(self) -> 'Tensor':
        return pypto.ceil(self)

    @source_location
    def floor(self) -> 'Tensor':
        return pypto.floor(self)

    @source_location
    def trunc(self) -> 'Tensor':
        return pypto.trunc(self)

    @source_location
    def reciprocal(self, precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor':
        return pypto.reciprocal(self, precision_type)

    @source_location
    def relu(self) -> 'Tensor':
        return pypto.relu(self)

    @source_location
    def permute(self, perm: List[int]) -> 'Tensor':
        return pypto.permute(self, perm)

    @source_location
    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        return pypto.transpose(self, dim0, dim1)

    @source_location
    def gather(self, dim: int, index: 'Tensor') -> 'Tensor':
        return pypto.gather(self, dim, index)

    @source_location
    def gathermask(self, pattern_mode: int) -> 'Tensor':
        return pypto.gathermask(self, pattern_mode)

    @source_location
    def index_add_(self, dim: int, index: 'Tensor', source: 'Tensor', *, alpha: Union[int, float] = 1) -> 'Tensor':
        pypto.index_add_(self, dim, index, source, alpha=alpha)
        return self

    @source_location
    def index_add(self, dim: int, index: 'Tensor', source: 'Tensor', *, alpha: Union[int, float] = 1) -> 'Tensor':
        return pypto.index_add(self, dim, index, source, alpha=alpha)

    @source_location
    def cumsum(self: 'Tensor', dim: int) -> 'Tensor':
        return pypto.cumsum(self, dim)

    @source_location
    def cumprod(self: 'Tensor', dim: int) -> 'Tensor':
        return pypto.cumprod(self, dim)

    @source_location
    def gcd(self: 'Tensor', other: 'Tensor | int') -> 'Tensor':
        return pypto.gcd(self, other)

    @source_location
    def triu(self: 'Tensor', diagonal: 'int | SymbolicScalar' = 0) -> 'Tensor':
        return pypto.triu(self, diagonal)

    @source_location
    def triu_(self: 'Tensor', diagonal: 'int | SymbolicScalar' = 0) -> 'Tensor':
        self.move(pypto.triu(self, diagonal))
        return self

    @source_location
    def tril(self: 'Tensor', diagonal: 'int | SymbolicScalar' = 0) -> 'Tensor':
        return pypto.tril(self, diagonal)

    @source_location
    def tril_(self: 'Tensor', diagonal: 'int | SymbolicScalar' = 0) -> 'Tensor':
        self.move(pypto.tril(self, diagonal))
        return self

    @source_location
    def expand_clone(
        self, shape: List[int], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
    ) -> 'Tensor':
        if valid_shape is None:
            valid_shape = []
        return pypto.expand_clone(self, shape, valid_shape=valid_shape)

    @source_location
    def pad(self, pad: Sequence[int], mode: str = "constant", value: Union[float, int] = 0) -> 'Tensor':
        return pypto.pad(self, pad, mode, value)

    @source_location
    def fillpad(self, mode: str = "constant", value: Union[float, int] = 0) -> 'Tensor':
        return pypto.fillpad(self, mode, value)

    @source_location
    def scatter_update(self, dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
        return pypto.scatter_update(self, dim, index, src)

    @source_location
    def scatter_(
        self, dim: int, index: 'Tensor', src: Union[float, Element, 'Tensor'], *, reduce: str = None
    ) -> 'Tensor':
        return pypto.scatter_(self, dim, index, src, reduce=reduce)

    @source_location
    def scatter(
        self, dim: int, index: 'Tensor', src: Union[float, Element, 'Tensor'], *, reduce: str = None
    ) -> 'Tensor':
        return pypto.scatter(self, dim, index, src, reduce=reduce)

    @source_location
    def var(
        self, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False
    ) -> 'Tensor':
        return pypto.var(self, dim, correction=correction, keepdim=keepdim)

    def _is_empty_slice(self, key):
        if isinstance(key, slice):
            return key.start is None and key.stop is None and key.step is None
        elif isinstance(key, (int, SymbolicScalar)):
            return False
        elif key is Ellipsis:
            return False
        return all([self._is_empty_slice(k) for k in key])

    def _normalize_key(self, key):
        if self._is_empty_slice(key):
            return key

        if isinstance(key, (int, SymbolicScalar, slice)) or key is Ellipsis:
            key = (key,)

        if not isinstance(key, tuple):
            raise FeError(RuntimeError("Invalid key type"))

        if any(k is Ellipsis for k in key):
            ellipsis_count = sum(k is Ellipsis for k in key)
            if ellipsis_count > 1:
                raise FeError(ValueError("Only one ... is supported"))

            ellipsis_pos = next(i for i, k in enumerate(key) if k is Ellipsis)
            other_len = len(key) - 1
            colon_count = self.dim - other_len
            if colon_count < 0:
                raise FeError(IndexError(f"Too many indices for tensor with dimension {self.dim}"))
            colons = (slice(None),) * colon_count
            key = key[:ellipsis_pos] + colons + key[ellipsis_pos + 1:]

        if len(key) < self.dim:
            missing_dims = self.dim - len(key)
            key += (slice(None),) * missing_dims

        if self.dim != len(key):
            raise FeError(IndexError(f"rank not match, expect {self.dim}, but got {len(key)}"))
        key = self._negative_index_to_positive(key, self.shape)
        return key


class ShmemTensor:
    def __init__(self):
        self._base = pypto_impl.ShmemTensor()

    @classmethod
    def from_base(cls, base: pypto_impl.ShmemTensor) -> 'ShmemTensor':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def base(self) -> pypto_impl.ShmemTensor:
        return self._base
