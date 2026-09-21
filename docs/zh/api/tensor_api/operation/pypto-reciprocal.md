# pypto.reciprocal

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

计算输入张量的元素级倒数，即`out = 1 / input`。

## 函数原型

```python
pypto.reciprocal(input, precision_type=pypto.PrecisionType.HIGH_PRECISION) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| input | Tensor | 输入张量。<br>支持的数据类型为：DT_FP16、DT_BF16、DT_FP32。<br>不支持空Tensor；支持的维度：1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | PrecisionType，可选 | 倒数操作的精度模式。默认值为`PrecisionType.HIGH_PRECISION`。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。<br>**INTRINSIC**：直接使用芯片指令进行计算，速度更快。 |

## 返回值说明

| 类型 | 说明 |
|:-----|:-----|
| Tensor | 包含输入张量元素级倒数的新张量。 |

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### 示例1：基本使用

```python
import pypto

x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x)

# Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
# Output y: [-2.1763, -0.4713, -0.6986,  1.3702]
```

### 示例2：使用高精度模式

```python
import pypto

# 使用高精度模式进行FP16计算
x = pypto.tensor([4], pypto.DT_FP16)
y = pypto.reciprocal(x, pypto.PrecisionType.HIGH_PRECISION)

# Input x:  [4]
# Output y: [0.25]
```

### 示例3：使用指令模式

```python
import pypto

# 使用指令模式
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x, pypto.PrecisionType.INTRINSIC)

# Input x:  [4]
# Output y: [0.25]
```

## 相关接口

- [pypto.rsqrt](pypto-rsqrt.md)：计算输入张量的元素级平方根的倒数。
- [pypto.div](pypto-div.md)：计算两个张量的元素级除法。
