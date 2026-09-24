# SqueezeMode

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

SqueezeMode定义了[vf.squeeze](../comparison_and_selection/squeeze.md)的数据收集模式，用于控制是否将有效元素的字节计数存储到AR SPR（Special Purpose Register）中。

## 原型定义

```python
class SqueezeMode(enum.Enum):
     NO_STORE_REG = ...  # 不将有效元素字节计数存入AR SPR（默认）
     STORE_REG = ...  # 将有效元素总字节数存入AR SPR
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.squeeze(src, preg, mode=pl.SqueezeMode.STORE_REG)
```
