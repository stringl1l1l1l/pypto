# SaturationMode

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

SaturationMode定义了浮点数转整数时的溢出处理方式，用于控制当源数据超出目标整数类型表示范围时的处理策略，确保转换结果的正确性和可预测性。

## 原型定义

```python
class SaturationMode(enum.Enum):
     OFF = ...   # 截断模式（默认），直接截断超出部分，可能导致溢出
     ON = ...    # 饱和模式，超出范围的值会被截断到目标类型的最大值或最小值
```
