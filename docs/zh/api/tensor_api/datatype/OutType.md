# OutType

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

OutType定义了输出数据的类型，用于指定某些操作（如比较操作）的输出格式，区分布尔值输出和位值输出。

## 原型定义

```python
class OutType(enum.Enum):
     BOOL = ...  # 布尔输出类型，输出True或False
     BIT = ...   # 位输出类型，输出0或1的位值
```
