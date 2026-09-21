# ReLuType

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

ReLuType定义了RELU激活函数的模式，用于启用RELU功能。

## 原型定义

```python
class ReLuType(enum.Enum):
     NO_RELU= ...  # 不使能ReLu功能
     RELU= ...     # 使能ReLu功能
```
