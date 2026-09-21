# pypto.gather

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

对输入的input，按照指定维度dim和索引index提取原始Tensor的对应值，最后返回结果。例如对3维Tensor，有以下计算公式：

$$
\begin{cases}
output[i,j,k] = input[index[i,j,k], j, k] & \text{if } dim = 0; \\
output[i,j,k] = input[i, index[i,j,k], k] & \text{if } dim = 1; \\
output[i,j,k] = input[i,j, index[i,j,k]] & \text{if } dim = 2.
\end{cases}
$$

## 函数原型

```python
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor，Shape支持1-5维，且shape size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定索引的维度。<br>支持任意合法的维度索引，范围为：-input.dim到input.dim - 1。 |
| index   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT32，DT_INT64。<br>不支持空Tensor，Shape支持1-4维，需保证index所有轴上的Shape大小不超过input的对应Shape大小，且值为合法索引，即不超过input在dim轴上的Shape大小。 |

## 返回值说明

返回输出Tensor，输出Tensor数据类型与input数据类型保持一致；输出Tensor的Shape与index的Shape相同。

## 约束说明

1. index.dim = input.dim，且index.shape\[i\] <= input.shape\[i\] (i != dim)，值为合法索引，即不能超出input.shape\[dim\]；

2. dim: -input.dim <= dim < input.dim；

3. input.shape的dim轴不可切，要求viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)，其余维度的Shape大小不做限制；

4. TileShape的维度与index相同，用于切分input和index，input的dim轴不可切，且所有输入和输出的TileShape大小总和不能超过UB内存的大小；

5. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT16，DT_INT32，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT16，DT_INT32，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT16，DT_INT32，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16
   <!-- end id6 -->

6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input为[x, y, z]，dim为1，输入index为[m, t, p]，输出为[m, t, p]，其中m <= x，p <= z，TileShape设置为[m1, t1, p1]，则m1，t1，p1分别用于切分m，t，p轴。y轴不可切，必须保证y轴全载。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

结果示例如下：

```python
输入数据x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
输出数据y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```
