# pypto.softmax

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

对输入Tensor在指定维度上应用softmax函数，将该维度的元素归一化为取值在 \[0, 1\] 之间的概率分布（所有元素之和为1），计算公式为：

$$
\text{softmax}(input)_i = \frac{e^{input_i - \text{max}(input)}}{\sum_{j=1}^{k} e^{input_j - \text{max}(input)}}
$$

## 函数原型

```python
softmax(input: Tensor, dim: int) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：DT_FP32<br>不支持空Tensor；Shape大小不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定归一化的维度。<br>支持负索引（如 -1表示最后一个维度）。<br>需在 [-input.dim, input.dim-1] 范围内。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，dim指定维度上的元素之和为1。

## 约束说明

无

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.softmax(x, -1)
```

结果示例如下：

```python
输入数据x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
输出数据y: [[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
```
