# pypto.lrelu

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

对输入张量逐元素应用Leaky ReLU（带泄漏的线性整流函数）激活函数。计算公式如下：

$$
\text{res}_i =
\begin{cases}
\text{input}_i & \text{if } \text{input}_i \geq 0 \\
\text{negative\_slope} \cdot \text{input}_i & \text{if } \text{input}_i < 0
\end{cases}
$$

其中`negative_slope`为负斜率参数，默认值为`0.01`。

## 函数原型

```python
lrelu(input: Tensor, negative_slope: Union[float, Element] = 0.01) -> Tensor
```

## 参数说明

| 参数名         | 输入/输出 | 说明                                                                 |
|----------------|-----------|----------------------------------------------------------------------|
| input          | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| negative_slope | 输入      | 负区间的斜率系数。<br>支持的类型为float/Element类型。默认值为`0.01`（float类型），当为float类型时会自动转换为Element类型，float对应DT_FP32。当需要使用其他数据类型时，可以通过Element构建。<br>必须为非负实数（≥ 0），不支持`nan`、`inf`等特殊值。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape与input一致。

## 约束说明

1. input数据类型必须为DT_FP16、DT_BF16或DT_FP32。
2. negative_slope必须为非负浮点数（≥ 0），且不能为`nan`或`inf`。
3. 不支持in-place操作（即输出不能与输入共享内存）。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([[-1.0, 0.0, 1.0]], pypto.DT_FP32)
out = pypto.lrelu(a)
```

结果示例如下：

```python
输入数据a:   [[-1.0  0.0  1.0]]
输出数据out: [[-0.01  0.0   1.0]]
```
