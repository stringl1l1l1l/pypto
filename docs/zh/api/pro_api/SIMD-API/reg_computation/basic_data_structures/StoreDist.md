# StoreDist

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

StoreDist定义了[vf.store_align](../data_movement/store_align.md)的数据存储分布模式，用于控制从寄存器到UB的数据搬运方式。

## 原型定义

```python
class StoreDist(enum.Enum):
     # reg_tensor单搬出模式
     NORM = ...  # 普通对齐存储（默认），根据dtype自动选择位宽粒度；64位宽数据类型DT_INT64/DT_UINT64只支持此模式
     NORM_B8 = ...  # 按8位宽类型普通存储
     NORM_B16 = ...  # 按16位宽类型普通存储
     NORM_B32 = ...  # 按32位宽类型普通存储
     FIRST_ELEMENT = ...  # 仅存储lane 0（首个元素），根据dtype自动选择位宽粒度
     FIRST_ELEMENT_B8 = ...  # 按8位宽类型仅存储首个元素
     FIRST_ELEMENT_B16 = ...  # 按16位宽类型仅存储首个元素
     FIRST_ELEMENT_B32 = ...  # 按32位宽类型仅存储首个元素
     PACK = ...  # 压缩存储，根据mask将src中有效元素的低半部分bit数据连续存储于dst中，根据dtype自动选择位宽粒度
     PACK_B16 = ...  # 按16位宽类型压缩存储
     PACK_B32 = ...  # 按32位宽类型压缩存储
     PACK_B64 = ...  # 按B64粒度压缩存储
     PACK4 = ...  # 4元素压缩存储，将有效元素的低8bit数据连续存储（32位宽类型）
     PACK4_B32 = ...  # 按32位宽类型4元素压缩存储
     # reg_tensor双搬出模式
     INTLV = ...  # 交错存储，将src0、src1中的元素交错存储（根据dtype自动选择位宽类型）
     INTLV_B8 = ...  # 按8位宽类型交错存储
     INTLV_B16 = ...  # 按16位宽类型交错存储
     INTLV_B32 = ...  # 按32位宽类型交错存储
     # mask_reg模式
     # NORM 同上，搬运VL/8数据
     # PACK 同上，每间隔1bit舍弃数据，将VL/8的数据压缩为VL/16搬出
```

## 约束说明

StoreDist的取值根据目标模式不同，支持的模式有所区别：

- **reg_tensor单搬出模式**：支持 NORM（含NORM_B8/NORM_B16/NORM_B32）、FIRST_ELEMENT（含FIRST_ELEMENT_B8/B16/B32）、PACK（含PACK_B16/B32/B64）、PACK4（含PACK4_B32）
- **reg_tensor双搬出模式**：支持 INTLV（含INTLV_B8/INTLV_B16/INTLV_B32）（需要两个源寄存器）
- **mask_reg模式**：支持 NORM、PACK（不支持显式粒度形式）

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, dist=pl.StoreDist.NORM)
```
