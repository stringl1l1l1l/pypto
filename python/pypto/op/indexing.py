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
"""PyPTO"""

from typing import Union

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..enum import ScatterMode
from ..error import PyptoError
from ..tensor import Tensor


@op_wrapper
def index_add_(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor:
    """
    Accumulate the elements of `alpha` times `source` into `input` tensor by
    adding to the indices in the order given in `index`.

    For a 3-D tensor this function specified output as:
    input[index[i], :, :] += alpha * source[i, :, :]  # if dim == 0
    input[:, index[i], :] += alpha * source[:, i, :]  # if dim == 1
    input[:, :, index[i]] += alpha * source[:, :, i]  # if dim == 2

    Parameters
    ----------
    input : Tensor
        Source tensor that needs to be added in place.
    dim : int
        Dimension along which to index. Negative indexing is supported.
    index : Tensor
        Indices of `source` to select from, should have dtype either int64
        or int32 and the dimension must be 1.The length of `index` must have
        the same size as the `dim` th dimension of `source`.
    source : Tensor
        The tensor containing values to add. The dimth dimension of
        `source` must have the same size as the length of `index`, and
        all other dimensions must match `self`, or an error will be raised.

    Keyword Arguments:
    ----------
    alpha : Number
        The scalar multiplier for `source`.

    Returns
    -------
    Tensor
        A new tensor sharing the same storage with the `input` tensor.

    Raises
    ------
    RuntimeError
        If any value in `index` is outside the inclusive range
        [0, source.shape[dim]-1].

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
    source = pypto.tensor([3, 3], pypto.DT_INT32)        # shape (3, 3)
    index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
    dim = 0

    # use alpha
    y = pypto.index_add_(x, dim, index, source, alpha=1)

    # not use alpha
    y = pypto.index_add_(x, dim, index, source)

    Input x:   [[0 0 0],
                [0 0 0]]
    source:    [[1 1 1],
                [1 1 1],
                [1 1 1]]
    index:      [0 1 0]

    Output y:  [[2 2 2],
                [1 1 1]]               # shape (2, 3)
    """

    pypto_impl.IndexAdd_(input, source, index, dim, pypto_impl.Element(input.dtype, alpha))
    return input


@op_wrapper
def index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor:
    """
    The out-of-place version of index_add_()
    """
    input0 = input
    pypto_impl.IndexAdd_(input0, source, index, dim, pypto_impl.Element(input.dtype, alpha))
    return input0


@op_wrapper
def index_add__ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor:
    """
    The version of index_add_() in ub
    """

    input.Move(pypto_impl.IndexAddUB(input, source, index, dim, pypto_impl.Element(input.dtype, alpha)))
    return input


@op_wrapper
def index_add_ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor:
    """
    The out-of-place version of index_add__ub()
    """

    return pypto_impl.IndexAddUB(input, source, index, dim, pypto_impl.Element(input.dtype, alpha))


@op_wrapper
def index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None:
    """
    Puts values from the tensor `values` into the tensor `input` using the
    indices specified in `indices`(which is a tuple of Tensors).

    With different numbers of tensors in indices, this function specified output as:
    input[indices[0][i], ...] = values[i, ...]                      # with 1 tensor in indices
    input[indices[0][i], indices[1][i], ...] = values[i, ...]       # with 2 tensors in indices
    input[indices[0][i], ..., indices[k][i], ...] = values[i, ...]  # with k tensors in indices

    Parameters
    ----------
    input : Tensor
        Source tensor that needs to be updated in place.
    indices : a tuple of 1-dimensional Tensor(s)
        The i-th 1-dimensional tensor represents the index along the
        i-th dimension in `input`, with a dtype of either int64 or int32.
        Broadcasting is not currently supported, and each 1-dimensional
        tensor must have the same length.
    values : Tensor
        Tensor of the same dtype as self. Broadcasting is not currently
        supported. The size of the first dimension of `values` must be
        the same as the length of the 1-dimensional tensors in `indices`.
        All other dimensions must match `input`.
    accumulate : bool
        Specify whether to accumulate into `input`. Specifically, when
        `indices` contain duplicate elements, the behavior is undefined.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If any value in the i-th 1-dimensional tensor of `indices` exceed
        the range [0, input.shape[i - 1]].

    Examples
    --------
    x = pypto.tensor([4, 2], pypto.DT_FP32)
    indices = (pypto.tensor([3], pypto.DT_INT32), )
    values = pypto.tensor([3, 2], pypto.DT_FP32)

    input x:  [[0 0],
               [0 0],
               [0 0],
               [0 0]]
      indices: [0 1 3]
       values: [[1 1],
               [2 2],
               [3 3]]

    updated x: [[1 1],
               [2 2],
               [0 0],
               [3 3]]
    """
    indices_list = list(indices)
    pypto_impl.IndexPut_(input, indices_list, values, accumulate)


@op_wrapper
def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gather elements from `input` along `dim` according to `index`.

    This function specified output for a 3-D tensor:
    output[i][j][k] = input[index[i][j][k]][j][k] # if dim == 0
    output[i][j][k] = input[i][index[i][j][k]][k] # if dim == 1
    output[i][j][k] = input[i][j][index[i][j][k]] # if dim == 2

    Parameters
    ----------
    input : Tensor
        Source tensor from which to gather values.
    index : Tensor
        Integer tensor containing the subscripts to pick along `dim`. It must have
        the same numbers of dimensions as `input`. And it is also required that
        index.shape[d] <= input.shape[d] for all dimensions d != dim.
    dim : int
        Dimension in `input` along which to gather. Negative indexing is supported.

    Returns
    -------
    Tensor
        A new tensor, with the same dtype as `input` and the same shape as `index`.

    Raises
    ------
    IndexError
        If any value in `index` is outside the inclusive range
        [0, input.shape[dim]-1].
    RuntimeError
        If the broadcast shape of `index` against `input` is incompatible.

    Examples
    --------
    x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)

    index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
    dim = 0
    y = pypto.gather(x, dim, index)

    Input x:  [[0 1 2 3 4],
               [5 6 7 8 9],
               [10 11 12 13 14]]
    index:    [[0 1 2 0],
               [1 2 0 1],
               [2 2 1 0]]

    Output y: [[0 6 12 3],
               [5 11 2 8],
               [10 11 7 3]]               # shape (3, 4)

    """

    return pypto_impl.GatherElements(input, index, dim)


@op_wrapper
def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gathers slices from `param` along a single axis `dim` using `indices`.
    param ∈ {S₀xS₁x…xS_{n-1}}, indices ∈ {I₀xI₁x…xI_{m-1}}.

    Output shape
    out.shape = (S₀,…,S_{dim-1}, I₀,…,I_{m-1}, S_{dim+1},…,S_{n-1}).
    That is, the dimension `S_dim` in `param` is replaced by the full shape of `indices`,
    while all other dimensions of `param` are preserved.

    For any multi-indices
    i = (i₀,…,i_{m-1}), t = (t₀,…,t_{n-2}),
    out[i, t] = param[t₀,…,t_{dim-1}, indices[i], t_{dim},…,t_{n-2}].
    Parameters
    ----------
    input : Tensor
    1-4-D tensor of shape (S0, S1, …, Sn-1) that provides the source values to gather from.

    index : Tensor (integer type)
    1-2-D integer tensor of shape (I0, I1); every entry must satisfy 0 ≤ value < S_dim.

    dim : int
    int axis in the range -n ≤ dim < n along which to gather; negative values are interpreted as dim + n.

    Examples
    --------
    x = pypto.tensor([3, 4], pypto.DT_FP32)
    indices = pypto.tensor([2,], pypto.DT_INT32)
    out0 = pypto.index_select(x, 0, indices)
    out1 = pypto.index_select(x, 1, indices)

    Input x:       [[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-0.4664,  0.2647, -0.1228, -1.1068],
                    [-1.1734, -0.6571,  0.7230, -0.6004]]
    Input indices:  [0, 2]
    Output out1 :  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-1.1734, -0.6571,  0.7230, -0.6004]]
    Output out2 :  [[ 0.1427, -0.5414],
                    [-0.4664, -0.1228],
                    [-1.1734,  0.7230]]
    """

    return pypto_impl.index_select(input, dim, index)


@op_wrapper
def scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Update 'input' in place with values from 'src' at the positions specified by 'index'.

    This function calculates the formula:
    For dim2, with s = index.shape[1],
    input[index[i][j]][:] = src[i * s + j][:]
    For dim4, with block_size = input.shape[1],
    input[index[i][j] // block_size][index[i][j] % block_size][0][:] = src[i][j][0][:]

    Parameters
    ----------
    input : Tensor
        The initialized destination tensor to update in place.
    dim : int
        Must be -2 for this specialized scatter operation.
    index : Tensor
        Global token indices. For 4-D input, each index addresses the flattened
        first two dimensions; indices are not broadcast across blocks.
    src : Tensor
        The source elements to scatter.

    Returns
    -------
    Tensor
        The updated input tensor. Non-index positions retain their existing values.
        Use move to write the result back to the same destination. To preserve
        the original input, clone it before calling the kernel and update the clone.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal 2.
        If the dimension of 'input' and 'src' is not equal 2 or 4.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    # Input values below are supplied by the caller; pypto.tensor does not initialize them.
    # dim2
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(4, 3)
    x.move(pypto.scatter_update(x, -2, y, z))

    Input x:[[1 2 3],
             [4 5 6],
             [7 8 9],
             [10 11 12],
             [13 14 15],
             [16 17 18],
             [19 20 21],
             [22 23 24]]
    Input y:[[1 2],
             [4 5]]
    Input z:[[1 2 3],
             [4 5 6],
             [7 8 9],
             [10 11 12]]
    Output x:[[1 2 3],
              [1 2 3],
              [4 5 6],
              [10 11 12],
              [7 8 9],
              [10 11 12],
              [19 20 21],
              [22 23 24]]

    #dim4
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(1, 2, 1, 3)
    x.move(pypto.scatter_update(x, -2, y, z))

    Input x:[[
                [[1 2 3]],
                [[4 5 6]],
                [[7 8 9]],
                [[10 11 12]],
                [[13 14 15]],
                [[16 17 18]],
             ],
             [
                [[19 20 21]],
                [[22 23 24]],
                [[25 26 27]],
                [[28 29 30]],
                [[31 32 33]],
                [[34 35 36]],
             ]]
    Input y:[[1 8],
             [4 10]]
    Input z:[[
                [[1 2 3]],
                [[4 5 6]],
             ],
             [
                [[7 8 9]],
                [[10 11 12]],
             ]]
    Output x:[[
                [[1 2 3]],
                [[1 2 3]],
                [[7 8 9]],
                [[10 11 12]],
                [[7 8 9]],
                [[16 17 18]],
             ],
             [
                [[19 20 21]],
                [[22 23 24]],
                [[4 5 6]],
                [[28 29 30]],
                [[10 11 12]],
                [[34 35 36]],
             ]]
    """
    if dim != -2:
        raise PyptoError(0xF00002, ValueError("scatter currection only support the case where dim = -2."))
    dims = input.Dim()
    if dims == 4:
        chunk_size = input.GetShape()[1]
    elif dims == 2:
        chunk_size = 1
    else:
        raise PyptoError(0xF00002, ValueError("dim must be 2 or 4"))

    return pypto_impl.ScatterUpdate(input, index, src, -2, "PA_BSND", chunk_size)


def get_scatter_mode(reduce: str):
    if reduce is None:
        return ScatterMode.NONE
    elif reduce == 'add':
        return ScatterMode.ADD
    elif reduce == 'multiply':
        return ScatterMode.MULTIPLY
    else:
        raise PyptoError(0xF00002, ValueError("scatter reduce only support 'add', 'multiply'"))


@op_wrapper
def scatter_(
    input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None
) -> Tensor:
    """Write all values from the value 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For a 3-D tensor, 'input' is update as:
    self[index[i][j][k]][j][k] = src  # if dim == 0
    self[i][index[i][j][k]][k] = src  # if dim == 1
    self[i][j][index[i][j][k]] = src  # if dim == 2

    Parameters
    ----------
    input : Tensor
        The input tensor to be chenged.
    dim : int
        The axis along which to index.
    index : Tensor
        The indices of elements to scatter.
    src : Tensor or Element
        The Tensor or Element to scatter.

    Returns
    -------
    Tensor
        A new tensor containing the elements of input after scatter.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal to the dimension of 'input'.
        If the index.size(d) > input.size(d)
        If the index.size(d) > src.size(d) when src is Tensor and d != dim
        If the value of 'input[i][j][k]' is bigger than the shape size of the dimension of 'input'.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    # dim2 and src is scalar
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)

    Input x:  [[0 0 0 0 0],
               [0 0 0 0 0],
               [0 0 0 0 0]]
    Input y:  [[1 2],
               [0 1]]
    Output o:  [[2.0 0   0 0 0],
                [2.0 2.0 0 0 0],
                [0   2.0 0 0 0]]
    """
    if index.dtype not in (pypto_impl.DT_INT32, pypto_impl.DT_INT64):
        raise PyptoError(0xF00001, TypeError(f"index tensor must be of int32 or int64, but got {index.dtype}"))
    scatter_mode = get_scatter_mode(reduce)
    if isinstance(src, (int, float)):
        input.Move(pypto_impl.Scatter(input, index, pypto_impl.Element(input.dtype, src), dim, scatter_mode))
        return input
    elif isinstance(src, pypto_impl.Element):
        input.Move(pypto_impl.Scatter(input, index, src, dim, scatter_mode))
        return input
    elif isinstance(src, pypto_impl.Tensor):
        input.Move(pypto_impl.Scatter(input, index, src, dim, scatter_mode))
        return input
    else:
        raise PyptoError(
            0xF00001, TypeError(f"Expected src to be int, float, Element, or Tensor, but got {type(src).__name__}")
        )


@op_wrapper
def scatter(
    input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None
) -> Tensor:
    """Out-of-place version of 'scatter_'."""
    if index.dtype not in (pypto_impl.DT_INT32, pypto_impl.DT_INT64):
        raise PyptoError(0xF00001, TypeError(f"index tensor must be of int32 or int64, but got {index.dtype}"))
    scatter_mode = get_scatter_mode(reduce)
    if isinstance(src, (int, float)):
        return pypto_impl.Scatter(input, index, pypto_impl.Element(input.dtype, src), dim, scatter_mode)
    elif isinstance(src, pypto_impl.Element):
        return pypto_impl.Scatter(input, index, src, dim, scatter_mode)
    elif isinstance(src, pypto_impl.Tensor):
        return pypto_impl.Scatter(input, index, src, dim, scatter_mode)
    else:
        raise PyptoError(
            0xF00001, TypeError(f"Expected src to be int, float, Element, or Tensor, but got {type(src).__name__}")
        )


@op_wrapper
def gathermask(self: Tensor, pattern_mode: int) -> Tensor:
    """
    Based on the built-in Mask selected by PatternMode,
    the positions in the self Tensor where the corresponding Bit is 1 form the output Tensor,
    and the values where the Bit is 0 are directly discarded.
    There are 7 modes for PatternMode:
    - PatternMode=1: Take the first element of every two elements in the last axis.
    - PatternMode=2: Take the second element of every two elements in the last axis.
    - PatternMode=3: Take the first element of every four elements in the last axis.
    - PatternMode=4: Take the second element of every four elements in the last axis.
    - PatternMode=5: Take the third element of every four elements in the last axis.
    - PatternMode=6: Take the fourth element of every four elements in the last axis.
    - PatternMode=7: Take all elements in the last axis.

    Parameters
    ----------
    self : Tensor
        Source tensor from which to gather values.
    pattern_mode : int
        Only supports 1 to 7.

    Returns
    -------
    Tensor
        A new tensor, with the same dtype as `self`, and the Shape of the output Tensor is as follows:
        - pattern_mode <= 2, the output shape's trailing axis is self.shape's trailing axis / 2,
            while other axes match the self shape.
        - When 2 < pattern_mode < 7, the output shape's trailing axis is self.shape's trailing axis divided by 4,
            while other axes remain consistent with the self shape.
        - pattern_mode = 7, output shape = self shape.
    Raises
    ------
    patterModeError
        If any value in `pattern_mode` is outside the inclusive range [1, 7].
    RuntimeError
        If 1 <= pattern_mode <= 2, self.shape[self.shape.size()-1] % 2 == 0.
        If 3 <= pattern_mode <= 6, self.shape[self.shape.size()-1] % 4 == 0.

    Examples
    --------
    x = pypto.tensor([3, 6], pypto.DT_INT32)        # shape (3, 6)
    pattern_mode = 1
    y = pypto.gathermask(x, pattern_mode)

    Self x:  [[0 1 2 3 4 5],
               [6 7 8 9 10 11],
               [12 13 14 15 16 17]]
    pattern_mode:  1,

    Output y: [[0 2 4],
               [6 8 10],
               [12 14 16]]               # shape (3, 3)

    """

    if 1 <= pattern_mode <= 6:
        from ..operation import _gathermask_checks, _view_input_ids, _view_input_names, _view_original_shapes
        self_id = self.Id() if hasattr(self, 'Id') else self._base.Id()
        orig_view_shape = _view_original_shapes.get(self_id)
        orig_input_id = _view_input_ids.get(self_id)
        orig_input_name = _view_input_names.get(self_id)
        if orig_view_shape is not None:
            _gathermask_checks.append({
                'view_tensor_id': self_id,
                'input_tensor_id': orig_input_id,
                'input_tensor_name': orig_input_name,
                'viewshape_last': orig_view_shape[-1] if orig_view_shape else None,
                'pattern_mode': pattern_mode,
            })

    return pypto_impl.GatherMask(self, pattern_mode)
