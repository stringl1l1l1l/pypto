# pypto.scatter\_

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

将src的值写入input中。写入位置由index指定。3维计算公式如下，其他维度以此类推：
<br>src为固定标量时：
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$$
<br>src为Tensor时：
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$$

## 函数原型

```python
scatter_(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_INT64。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定用于索引的维度，支持input的维度范围内的任意维度。<br>合法的维度索引，范围为：-input.dim到input.dim - 1。 |
| index   | 输入      | input的一组索引。<br>支持的数据类型为：Tensor。<br>Tensor支持的数据类型为：INT64、INT32。<br>支持的维度：和input保持一致<br>对于所有d != dim的维度，需满足要求：index.size(d) <= input.size(d)<br>当src为Tensor时，所有维度都需满足：index.size(d) <= src.size(d)<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX） |
| src     | 输入      | src是更新的标量或Tensor。<br>src为Element时，支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_INT64，不支持输入INF/NAN<br>src为Tensor时，支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_INT64，数据类型和input保持一致。<br> |
| reduce  | 输入      | 要应用的归约操作，支持'add'或'multiply'，不传参时默认为直接替换 |

## 返回值说明

返回更新后的input，为inplace操作。

## 约束说明

1. broadcast约束：input和index不支持broadcast；

2. input.shape的dim轴不可切，viewshape的维度与input维度相同，要求viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)，其余维度的Shape大小不做限制；

3. input.shape的dim轴不可切，tileshape的维度与input维度相同，tileshape\[dim\] \>= viewshape\[dim\]，其余维度的Shape大小不做限制，input index和result都会在UB中，需满足所有输入和输出的tileshape大小总和不能超过UB内存的大小。

4. input.shape和index.shape的非dim轴切分，需满足viewshape[non dim]切分后，input和index的非dim轴切分块数相同。tileshape切分时，也需要保证input和index的非dim轴切分块数相同。

5. src为Tensor，reduce为None，且当index中存在指向相同位置的不唯一索引时，行为是不确定的，将从src中任意选择一个值写入。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[a, b, c]，dim为1，index为[m, t, p](其中m<=a, p<=c)，src为[x, y, z](其中x>=m, y>=t, z>=p)，输出为[a, b, c]，TileShape设置为[m1, t1, p1]。则m1，p1分别用于切分m，p轴。t1必须大于等于b和t，dim对应轴不可切，必须保证b轴和t轴全载。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

- 将2维input根据2维index更新对应索引的值

    ```python
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)
    ```

    结果示例如下：

    ```txt
    输入数据x:[[0 0 0 0 0],
               [0 0 0 0 0],
               [0 0 0 0 0]]
    输入数据y:[[1 2],
               [0 1]]
    输出数据o:[[2.0 0   0 0 0],
               [2.0 2.0 0 0 0],
               [0   2.0 0 0 0]]
    ```
