# SaturationFlagMode

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

SaturationFlagMode定义了[pypto_pro.language.set_saturation_flag](../special_reg_access/set_saturation_flag.md)和[pypto_pro.language.get_saturation_flag](../special_reg_access/get_saturation_flag.md)操作的饱和模式类别，用于控制CTRL寄存器中不同计算场景的饱和行为。

每个模式对应CTRL寄存器中的一个特定位：

| 枚举值 | CTRL寄存器位 | 控制范围 | 极性 |
|---|---|---|---|
| FLOAT | bit 48 | 浮点数计算和浮点数精度转换 | 反转（bit=0为饱和开启） |
| FLOAT8 | bit 50 | 浮点8计算 | 反转（bit=0为饱和开启） |
| INT | bit 53 | 整数计算 | 正常（bit=1为饱和开启） |
| CAST | bit 59 | 浮点转整数或整数转整数的精度转换 | 反转（bit=0为饱和开启） |

## 原型定义

```python
class SaturationFlagMode(enum.Enum):
    FLOAT = ...    # 浮点数计算和浮点数精度转换（CTRL bit 48）
    FLOAT8 = ...   # 浮点8计算（CTRL bit 50）
    INT = ...      # 整数计算（CTRL bit 53）
    CAST = ...     # 浮点转整数/整数转整数精度转换（CTRL bit 59）
```

## 约束说明

- 与[vf.astype](../type_conversion/astype.md)的[SaturateMode](SaturateMode.md)参数配合使用时，当CTRL[60]=0时为单指令模式（由saturate参数控制），当CTRL[60]=1时为全局模式（由本枚举控制的CTRL位生效）。

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def kernel(...):
    # 在VF计算前设置CAST类别的饱和模式为全局开启
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    # ... 执行VF计算 ...
    # 恢复
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)
```
