# pypto_pro.language.CacheLine

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

DCCI缓存操作范围枚举，用于指定对单个缓存行或整个数据缓存执行清理并失效操作。

## 原型定义

```python
PYPTO_DECLARE_ENUM(CacheLine,
    SINGLE_CACHE_LINE,   # 目标有效地址所在的64字节缓存行，地址无需手动对齐。
    ENTIRE_DATA_CACHE    # 操作整个数据缓存。
)
```

具体使用约束，请参见[dcci](../cache_control/dcci.md)。
