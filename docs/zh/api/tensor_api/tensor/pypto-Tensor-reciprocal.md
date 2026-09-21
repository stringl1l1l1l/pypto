# pypto.Tensor.reciprocal

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

## 函数原型

```python
reciprocal(self, precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | 输入      | 精度类型。 <br> 支持的类型为：PrecisionType。 <br> 默认值为PrecisionType.HIGH_PRECISION。 <br> HIGH_PRECISION使用更高精度的计算以减少精度损失；INTRINSIC直接使用芯片指令进行计算，速度更快。 |

## 详细说明

请参见[pypto.reciprocal](../operation/pypto-reciprocal.md)。
