# pypto.var

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

计算输入Tensor中dim维度所有数据的方差。计算公式为：
$$
\sigma^2 = \frac{1}{\max(0, ~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2
$$

## 函数原型

```python
var(input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False)
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|------------|---------- |----------------------------------------------------------------------|
| input      | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim        | 输入      | 指定进行归约的维度。<br>支持任意单轴或多轴。<br>默认为None，即全轴。 |
| correction | 输入      | 校正系数。<br>样本大小与样本自由度之间的差值。<br>默认为贝塞尔校正，即correction=1。 |
| keepdim    | 输入      | 归约后是否保留被归约的维度，默认False。 |

## 返回值说明

返回Tensor类型。其数据类型与输入Tensor一致。

keepdim为True时，对应dim的shape规约为1，其他轴的shape不变；keepdim为False时，对应dim会被移除。

## 约束说明

1. input.shape的dim轴不可切，viewshape的维度与input维度相同，要求viewshape\[dim\] \== input.shape\[dim\]，其余维度的Shape大小不做限制
2. dim中不支持重复值，且len(dim) <= input.dim
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

示例1：输入input shape为[m, n]，输出为[m, 1]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.var(x, 1, correction=1, keepdim=True)
```

结果示例如下：

```txt
输入数据x: [[1., 2., 3.],
            [4., 5., 6.]]
输出数据y: [[1.],
            [1.]]
```
