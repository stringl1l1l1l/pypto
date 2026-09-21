# MemBarMode

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

MemBarMode定义了[vf.mem_bar](../data_movement/mem_bar.md)内存屏障的源->目标排序约束，用于确保指定的前序访存操作在后序访存操作开始之前完成。

## 原型定义

```python
class MemBarMode(enum.Enum):
     VST_VLD = ...  # Vector Store -> Vector Load（RAW，默认）
     VLD_VST = ...  # Vector Load -> Vector Store（WAR）
     VST_VST = ...  # Vector Store -> Vector Store（WAW）
     VST_LD = ...  # Vector Store -> Scalar Load
     VST_ST = ...  # Vector Store -> Scalar Store
     VLD_ST = ...  # Vector Load -> Scalar Store
     ST_VLD = ...  # Scalar Store -> Vector Load
     ST_VST = ...  # Scalar Store -> Vector Store
     LD_VST = ...  # Scalar Load -> Vector Store
     VV_ALL = ...  # 所有Vector -> 所有Vector
     VS_ALL = ...  # 所有Vector -> 所有Scalar
     SV_ALL = ...  # 所有Scalar -> 所有Vector
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # 确保Vector Store在Vector Load之前完成
```
