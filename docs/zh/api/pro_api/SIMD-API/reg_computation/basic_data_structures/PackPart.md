# PackPart

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

PackPart定义了vf.pack和vf.unpack的半区选择，用于指定打包或解包操作处理的是寄存器的上半区还是下半区。

## 原型定义

```python
class PackPart(enum.Enum):
     LOWER = ...  # 下半区（默认）
     UPPER = ...  # 上半区（仅RegTraitNumTwo支持LOWER）
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.pack(src0, src1, preg, part=pl.PackPart.LOWER)
```
