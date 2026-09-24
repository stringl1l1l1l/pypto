# DataCopyMode

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

DataCopyMode定义了[vf.load_align](../data_movement/load_align.md)、[vf.store_align](../data_movement/store_align.md)和[vf.gather](../data_movement/gather.md)的数据拷贝粒度模式，用于控制是逐元素拷贝还是按DataBlock块拷贝。

## 原型定义

```python
class DataCopyMode(enum.Enum):
     NORM = ...  # 普通逐元素拷贝（默认）
     DATA_BLOCK_LOAD = ...  # DataBlock加载（用于vf.gather：按32B DataBlock粒度gather）
     DATA_BLOCK_COPY = ...  # 非连续DataBlock拷贝（基于block_stride）
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY)
```
