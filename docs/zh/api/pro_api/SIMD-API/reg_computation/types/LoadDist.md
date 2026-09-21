# LoadDist

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

LoadDist定义了[vf.load_align](../data_movement/load_align.md)的数据加载分布模式，用于控制从UB到寄存器的数据搬运方式。不同模式对应不同的硬件指令，影响数据在寄存器中的排布和广播/采样行为。

## 原型定义

```python
class LoadDist(enum.Enum):
     # RegTensor目标
     NORM = ...  # 普通逐元素加载（默认）
     BRC = ...  # 广播单个元素到整个寄存器
     BRC_B8 = ...  # 按8位宽类型广播
     BRC_B16 = ...  # 按16位宽类型广播
     BRC_B32 = ...  # 按32位宽类型广播
     US = ...  # 上采样（每个元素重复两次）
     US_B8 = ...  # 按8位宽类型上采样
     US_B16 = ...  # 按16位宽类型上采样
     DS = ...  # 下采样（每隔一个元素丢弃）
     DS_B8 = ...  # 按8位宽类型下采样
     DS_B16 = ...  # 按16位宽类型下采样
     UNPK = ...  # 解包
     UNPK_B8 = ...  # 按8位宽类型解包
     UNPK_B16 = ...  # 按16位宽类型解包
     UNPK_B32 = ...  # 按32位宽类型解包
     UNPK4 = ...  # 4元素解包
     BLK = ...  # 块拷贝
     E2B = ...  # 16位宽类型->32位宽类型扩展
     E2B_B16 = ...  # 按16位宽类型扩展
     E2B_B32 = ...  # 按32位宽类型粒度扩展
     DINTLV_B8 = ...  # 按8位宽类型去交错（拆分奇偶寄存器）
     DINTLV_B16 = ...  # 按16位宽类型去交错
     DINTLV_B32 = ...  # 按32位宽类型去交错
     # MaskReg目标
     # NORM/US/DS 同RegTensor目标，搬运数据量为VL/8字节
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    reg = vf.load_align(ub_tile, dist=pl.LoadDist.BRC)
```
