# pypto.prelu

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

对input的每个元素进行带参数的整流线性单元运算，当元素值大于等于0时保持不变，小于0时乘以权重系数。计算公式如下：

$$
res_i = \begin{cases}
input_i & \text{if } input_i \geq 0 \\
weight_i \times input_i & \text{if } input_i < 0
\end{cases}
$$

其中weight为一维张量：

- 当input为1维时，weight长度为1，按元素共享权重
- 当input为2-4维时，weight长度与input的第二维（通道维）大小相同，按通道共享权重

## 函数原型

```python
prelu(input: Tensor, weight: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16、DT_FP32、DT_BF16。<br>不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| weight  | 输入      | 权重参数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16、DT_FP32、DT_BF16，需与input类型相同。<br>Shape为一维，当input为1维时长度为1；当input为2-4维时长度与input的第二维大小相同。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和Shape与input相同。

## 约束说明

1. input和weight类型应该相同。
2. weight的Shape必须为一维，当input为1维时长度为1；当input为2-4维时长度等于input的第二维大小。
3. 由于存在临时内存使用，输入维度为二维时，TileShape大小有额外约束，假设TileShape为\[a,b\]，那么a*b*sizeof(self) + b/8 + 8KB < UB。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1（1D输入）：输入input与weight，shape分别为[n] [1]。输出为[n]，TileShape设置为[n1]，则n1用于切分n轴。

示例2（2D输入）：输入input与weight，shape分别为[m, n] [n]。输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

示例1：1D输入

```python
# 示例1：1D PReLU运算
# input shape为[4]，weight shape为[1]
# 对于负数元素，所有元素共享同一个权重
input_tensor = pypto.tensor([-2.0, 1.0, -3.0, 0.5], pypto.DT_FP32)
weight_tensor = pypto.tensor([0.25], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

结果示例如下：

```python
输入数据input:  [-2.0,  1.0, -3.0,  0.5]
输入数据weight: [ 0.25 ]
输出数据out:    [-0.5,  1.0, -0.75,  0.5]
```

计算过程说明：

- 对于所有元素，共享权重0.25
- -2.0 < 0，结果 = 0.25 × (-2.0) = -0.5；1.0 ≥ 0，结果 = 1.0
- -3.0 < 0，结果 = 0.25 × (-3.0) = -0.75；0.5 ≥ 0，结果 = 0.5

示例2：2D输入

```python
# 示例2：2D PReLU运算
# input shape为[2, 3]，weight shape为[3]
# 对于负数元素，按通道乘以对应权重
input_tensor = pypto.tensor([[-2.0, 1.0, -3.0], [0.5, -1.0, 2.0]], pypto.DT_FP32)
weight_tensor = pypto.tensor([0.25, 0.5, 0.1], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

结果示例如下：

```python
输入数据input:  [[-2.0,  1.0, -3.0], [ 0.5, -1.0,  2.0]]
输入数据weight: [ 0.25, 0.5,  0.1]
输出数据out:    [[-0.5,  1.0, -0.3], [ 0.5, -0.5,  2.0]]
```

计算过程说明：

- 第0通道：-2.0 < 0，结果 = 0.25 × (-2.0) = -0.5；0.5 ≥ 0，结果 = 0.5
- 第1通道：1.0 ≥ 0，结果 = 1.0；-1.0 < 0，结果 = 0.5 × (-1.0) = -0.5
- 第2通道：-3.0 < 0，结果 = 0.1 × (-3.0) = -0.3；2.0 ≥ 0，结果 = 2.0
