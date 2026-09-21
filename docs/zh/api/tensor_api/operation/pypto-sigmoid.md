# pypto.sigmoid

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

对输入Tensor的每个元素应用sigmoid激活函数，计算公式为：

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

## 函数原型

```python
sigmoid(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：DT_FP32。<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入元素经sigmoid函数映射到 \(0, 1\)区间的结果。

## 调用示例

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sigmoid(x)
```

结果示例如下：

```python
输入数据x: [-3.0, 0.0, 2.0, 5.0]
输出数据y: [0.0474, 0.5000, 0.8808, 0.9933]
```
