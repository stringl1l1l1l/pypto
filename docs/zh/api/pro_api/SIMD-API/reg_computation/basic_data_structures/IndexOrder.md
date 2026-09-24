# IndexOrder

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

IndexOrder定义了[vf.arange](../sorting_and_indexing/arange.md)生成的索引序列方向，用于控制序列是递增还是递减。

## 原型定义

```python
class IndexOrder(enum.Enum):
     INCREASE_ORDER = ...  # 递增序列：dst[i] = start + i（默认）
     DECREASE_ORDER = ...  # 递减序列：dst[i] = start - i
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.arange(start=0, order=pl.IndexOrder.INCREASE_ORDER)
```
