# MaskPattern

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

MaskPattern定义了[vf.create_mask](../mask_operations/create_mask.md)和[vf.mask_gen_with_reg_tensor](../mask_operations/mask_gen_with_reg_tensor.md)的掩码生成模式，用于控制寄存器中哪些元素被标记为有效。

## 原型定义

```python
class MaskPattern(enum.Enum):
     ALL = ...  # 所有元素有效
     ALLF = ...  # 所有元素无效
     VL1 = ...  # 最低1个元素有效
     VL2 = ...  # 最低2个元素有效
     VL3 = ...  # 最低3个元素有效
     VL4 = ...  # 最低4个元素有效
     VL8 = ...  # 最低8个元素有效
     VL16 = ...  # 最低16个元素有效
     VL32 = ...  # 最低32个元素有效
     VL64 = ...  # 最低64个元素有效
     VL128 = ...  # 最低128个元素有效
     M3 = ...  # 每3个元素中第1个有效
     M4 = ...  # 每4个元素中第1个有效
     H = ...  # 低半部分有效
     Q = ...  # 低四分之一有效
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.VL8)  # 最低8个元素有效
```
