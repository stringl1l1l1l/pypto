# pypto.Tensor.fmod

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
fmod(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 除数。 <br> 支持的类型为：Tensor、int、float。 |
| precision_type | 输入 | 精度模式枚举类型，用以控制取模计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md)。<br> 默认为HIGH_PRECISION（高精度模式）。 |

## 详细说明

请参见[pypto.fmod](../operation/pypto-fmod.md)。
