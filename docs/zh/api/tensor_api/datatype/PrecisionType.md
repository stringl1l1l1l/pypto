# PrecisionType

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

PrecisionType定义了高精度向量算子的精度模式，用于控制除法、取模、指数、对数等计算时的精度处理方式。

## 原型定义

```python
class PrecisionType(enum.Enum):
    INTRINSIC = ...       # 指令模式，直接使用芯片指令
    HIGH_PRECISION = ...  # 高精度模式
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| HIGH_PRECISION | 高精度模式。在底层实现中使用更高精度的计算方式。 |
| INTRINSIC | 指令模式。直接使用芯片指令进行计算，性能更高。 |

## 使用建议

1. **默认行为**：如果不指定精度模式，默认使用`HIGH_PRECISION`模式，以确保计算精度。
2. **精度要求高的场景**：推荐使用`HIGH_PRECISION`模式，可以有效减少精度损失，提高计算结果的准确性。
3. **对精度要求不高但追求性能的场景**：可以使用`INTRINSIC`模式，直接使用芯片指令进行计算。
4. HIGH_PRECISION使用说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：支持
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：不支持
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：不支持
   <!-- end id6 -->

## 支持的算子

以下算子支持`PrecisionType`参数：

| 算子 | 说明 |
|:-----|:-----|
| div | 元素级除法 |
| fmod | 元素级取模 |
| remainder | 元素级余数 |
| pow | 元素级幂运算 |
| exp | 指数运算 |
| sqrt | 开方运算 |
| rsqrt | 开方倒数运算 |
| log | 对数运算 |
| log2 | 以2为底的对数运算 |
| log10 | 以10为底的对数运算 |
| reciprocal | 倒数运算 |

## 使用示例

```python
import pypto

# 创建张量
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)

# 使用高精度模式
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)

# 使用指令模式
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)

# 默认使用高精度模式
out = pypto.div(a, b)

# 使用运算符（自动使用高精度模式）
out = a / b

# 其他算子示例
out = pypto.exp(a, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.sqrt(a, pypto.PrecisionType.INTRINSIC)
out = pypto.log(a, pypto.LogBaseType.LOG_E, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.pow(a, b, pypto.PrecisionType.HIGH_PRECISION)
```
