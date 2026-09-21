# ScatterMode

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

ScatterMode定义了scatter函数reduce模式

## 原型定义

```python
class ScatterMode(enum.Enum):
     NONE = ...     # 仅做数据搬运
     ADD = ...      # 加法模式
     MULTIPLY = ... # 乘法模式
```
