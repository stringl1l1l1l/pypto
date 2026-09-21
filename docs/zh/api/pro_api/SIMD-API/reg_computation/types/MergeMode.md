# MergeMode

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

MergeMode定义了VF计算指令中mask未选中元素（非活跃元素）在目标寄存器中的处理方式。适用于[vf.add](../basic_arithmetic/add.md)、[vf.sub](../basic_arithmetic/sub.md)、[vf.mul](../basic_arithmetic/mul.md)、[vf.div](../basic_arithmetic/div.md)等VF计算接口。

## 原型定义

```python
class MergeMode(enum.Enum):
     ZEROING = ...  # mask未选中位置置零（默认）
     MERGING = ...  # mask未选中位置保留目标寄存器原值
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.add(src0, src1, preg, mode=pl.MergeMode.ZEROING)
```
