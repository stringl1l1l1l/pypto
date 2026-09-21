# MaskWidth

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

MaskWidth定义了[vf.get_mask_spr](../data_movement/get_mask_spr.md)读取SPR掩码时的位宽展开模式，用于控制从MASK寄存器读取掩码数据时的位扩展粒度。

## 原型定义

```python
class MaskWidth(enum.Enum):
     B32 = ...  # 读取64位MASK0，每位展开为4位（movp_b32，默认）
     B16 = ...  # 读取128位{MASK1,MASK0}，每位展开为2位（movp_b16）
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.get_mask_spr(width=pl.MaskWidth.B32)
```
