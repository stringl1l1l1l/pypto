# pypto.exp

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

计算输入Tensor中每个元素的e的指数，逐元素运算，返回与输入形状相同的Tensor。

## 函数原型

```python
pypto.exp(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| input | Tensor | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。<br>不支持空Tensor；支持的维度：1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | PrecisionType，可选 | 指数操作的精度模式。默认值为`PrecisionType.INTRINSIC`。<br>**INTRINSIC**：直接使用芯片指令进行计算，速度更快。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape与input相同。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp(x)
```

结果示例如下：

```python
输入数据x: [0.0    1.0    2.0]
输出数据y: [1.0000  2.7183  7.3891]
```

### 高精度模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.INTRINSIC)
```
