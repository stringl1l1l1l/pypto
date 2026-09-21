# CastLayout

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

CastLayout定义了[vf.astype](../type_conversion/astype.md)、[vf.exp_sub](../composite_computation/exp_sub.md)和[vf.muls_cast](../composite_computation/muls_cast.md)中源操作数和目的操作数位宽不同时，位宽小的元素在寄存器中的排布方式。

单条指令计算量以位宽更大的数据类型为准，layout用于控制位宽小的元素在寄存器中的排布位置。

## 原型定义

```python
class CastLayout(enum.Enum):
     ZERO = ...  # 结果写入偶数半区（PART_EVEN）
     ONE = ...  # 结果写入奇数半区（PART_ODD）
     TWO = ...  # 第三半区（FP4类型4x扩展/缩窄时使用）
     THREE = ...  # 第四半区（FP4类型4x扩展/缩窄时使用）
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT8, layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
```
