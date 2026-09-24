# DuplicatePos

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

DuplicatePos定义了[vf.full](../data_movement/full.md)广播模式下元素的复制位置选择，用于指定将哪个索引位置的元素广播到整个寄存器。

## 原型定义

```python
class DuplicatePos(enum.Enum):
     LOWEST = ...  # 广播最低索引位置的元素（默认）
     HIGHEST = ...  # 广播最高索引位置的元素
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.full(preg, pos=pl.DuplicatePos.LOWEST)
```
