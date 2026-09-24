# HistType

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

HistType定义了[vf.histograms](../sorting_and_indexing/histograms.md)直方图的统计模式，用于指定是累加统计还是频次统计。

## 原型定义

```python
class HistType(enum.Enum):
     ACCUMULATE = ...  # 累加模式（默认），统计结果累加到目标寄存器已有值上
     FREQUENCY = ...  # 频次模式，统计每个索引值的出现次数
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, hist_type=pl.HistType.FREQUENCY)
```
