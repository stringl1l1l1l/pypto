# pypto_pro.language.CrossCoreSyncMode

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

核间同步模式枚举，用于指定[set_cross_core](../synchronization/set_cross_core.md)和[wait_cross_core](../synchronization/wait_cross_core.md)参与同步的核以及SET/WAIT信号的配对方式。

## 原型定义

```python
PYPTO_DECLARE_ENUM(CrossCoreSyncMode,
    INTER_BLOCK,
    INTER_SUBBLOCK,
    INTRA_BLOCK,
    UNICAST_BLOCK
)
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| INTER_BLOCK | 模式值为0。多个AI Core之间的同类核全核同步。AIC场景同步本次Kernel启动的所有AIC，AIV场景同步本次Kernel启动的所有AIV；AIC与AIV不会在该模式下互相同步。 |
| INTER_SUBBLOCK | 模式值为1。同一AI Core内的AIV0与AIV1同步，不同AI Core之间互不影响。 |
| INTRA_BLOCK | 模式值为2。同一AI Core内的AIC与全部AIV同步。AIV到AIC方向须由AIV0和AIV1分别发送信号，AIC等待两路信号；AIC到AIV方向由AIC发送信号，AIV0和AIV1分别等待。该值为set_cross_core和wait_cross_core的默认同步模式。 |
| UNICAST_BLOCK | 模式值为3。同一AI Core内的AIC与单个AIV同步。AIC侧事件ID 0～15对应AIV0，16～31对应AIV1；AIV侧事件ID取0～15。 |

具体事件ID范围、流水限制及各模式的SET/WAIT配对要求，请参见[set_cross_core](../synchronization/set_cross_core.md)和[wait_cross_core](../synchronization/wait_cross_core.md)。
