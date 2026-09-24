# BinType

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

BinType定义了[vf.histograms](../sorting_and_indexing/histograms.md)直方图统计的索引区间范围，用于指定统计的是源寄存器中哪一部分索引区间的元素分布。

## 原型定义

```python
class BinType(enum.Enum):
     BIN0 = ...  # 低半区索引范围 [0-127]（默认）
     BIN1 = ...  # 高半区索引范围 [128-255]
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, bin_type=pl.BinType.BIN0)
```
