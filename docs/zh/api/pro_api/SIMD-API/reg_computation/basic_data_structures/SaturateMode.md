# SaturateMode

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

SaturateMode定义了[vf.astype](../type_conversion/astype.md)缩窄类型转换时的饱和处理模式，用于控制源数据超出目标类型表示范围时的处理策略。

## 原型定义

```python
class SaturateMode(enum.Enum):
     OFF = ...  # 非饱和模式（默认），超出范围的值截断或行为因转换场景而异
     ON = ...  # 饱和模式，超出范围的值截断到目标类型的最大值或最小值
```

## 约束说明

不同类型转换场景下，非饱和模式与饱和模式的行为差异如下：

| 场景 | 非饱和模式（OFF） | 饱和模式（ON） |
|---|---|---|
| 浮点转整数 | 输入数据超过输出类型最值时，结果被截断为目标格式的数据宽度（保留最低有效位），例如输入half值为257，输出uint8_t值为1；输入为+/-inf时，则返回输出类型的对应最值；输入为nan时，返回0。 | 输入数据超过输出类型最值时，返回输出类型的对应最值，例如输入half值为257，输出uint8值为255，输入half值为-inf，输出uint8_t值为0；输入为nan时，返回0。 |
| 浮点转浮点 | 输入数据为nan时，输出为nan；输入+/-inf时，输出为+/-inf。 | 输入为nan时，输出为0；输入数据超过输出类型最值时，返回输出类型的对应最值。 |
| 整数转浮点 | 不支持非饱和模式 | 输入为nan时，输出为0；输入数据超过输出类型最值时，返回输出类型的对应最值。该场景默认饱和模式，无需配置。 |
| 整数转整数 | 输入数据会截断为目标数据宽度，例如，输入int32_t值为256，输出uint8_t值为0。 | 输入数据超出目标数据范围，会饱和为目标数据最值。 |

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT8, saturate=pl.SaturateMode.ON)
```
