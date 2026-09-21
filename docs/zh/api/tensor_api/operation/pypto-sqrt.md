# pypto.sqrt

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

计算输入Tensor中每个元素的平方根，逐元素运算。输入为负数时返回NaN。

## 函数原型

```python
pypto.sqrt(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| input | Tensor | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为DT_FP16、DT_BF16、DT_FP32。<br>不支持空Tensor；支持的维度：1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | PrecisionType，可选 | 平方根操作的精度模式。默认值为`PrecisionType.INTRINSIC`。<br>**INTRINSIC**：直接使用芯片指令进行计算，速度更快。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入Tensor对应元素的平方根。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sqrt(x)
```

结果示例如下：

```txt
输入数据x: [1.0, 4.0, 9.0, 16.0, 25.0]
输出数据y: [1.0, 2.0, 3.0, 4.0,  5.0]
```

### 高精度模式示例

```python
x = pypto.tensor([5], pypto.DT_FP16)
y = pypto.sqrt(x, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
x = pypto.tensor([5], pypto.DT_FP16)
y = pypto.sqrt(x, pypto.PrecisionType.INTRINSIC)
```
