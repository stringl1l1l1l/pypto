# pypto.log2

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

对input做以2为底的对数运算

## 函数原型

```python
pypto.log2(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| input | Tensor | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>支持的维度：1-4维<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | PrecisionType，可选 | 对数操作的精度模式。默认值为`PrecisionType.INTRINSIC`。<br>**INTRINSIC**：直接使用芯片指令进行计算，速度更快。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为input大小。

## 约束说明

1. 输入Tensor和输出Tensor类型应该相同。
2. 输入元素需大于0，否则输出为NaN。

## TileShape设置示例

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

### 接口调用示例

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log2(x)
```

结果示例如下：

```python
输入数据x: [1.0     2.0     3.0]
输出数据y: [0.0000 1.0000 1.5849]
```

### 高精度模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.INTRINSIC)
```
