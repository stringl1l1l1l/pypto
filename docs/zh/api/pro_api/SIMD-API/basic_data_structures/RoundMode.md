# pypto_pro.language.RoundMode

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

类型转换接口使用的舍入模式枚举。不同接口及数据类型转换组合支持的舍入模式可能不同，具体以对应接口的参数范围或约束说明为准。

## 原型定义

```python
class RoundMode(enum.Enum):
    CAST_NONE = ...
    CAST_RINT = ...
    CAST_ROUND = ...
    CAST_FLOOR = ...
    CAST_CEIL = ...
    CAST_TRUNC = ...
    CAST_ODD = ...
```

## 参数说明

| 枚举值 | 说明 | 示例 |
|---|---|---|
| CAST_NONE | 不显式指定舍入规则，具体转换行为由使用该枚举的接口定义 | - |
| CAST_RINT | 舍入到最近值，中间值取偶数 | 2.5 → 2，3.5 → 4 |
| CAST_ROUND | 舍入到最近值，中间值远离零 | 2.5 → 3，-2.5 → -3 |
| CAST_FLOOR | 向负无穷方向舍入 | 1.6 → 1，-1.6 → -2 |
| CAST_CEIL | 向正无穷方向舍入 | 1.6 → 2，-1.6 → -1 |
| CAST_TRUNC | 向零方向舍入 | 1.6 → 1，-1.6 → -1 |
| CAST_ODD | 发生精度丢失时，将结果的最低有效位设为1 | DT_FP32转DT_FP16：1.0001 → 1.0009765625 |

## 约束说明

- 各接口的默认舍入模式可能不同。
- 不同数据类型转换组合仅支持部分枚举值，使用前应查阅对应接口文档。
