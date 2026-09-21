# pypto_pro.language.DcciDst

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

DCCI一致性目标枚举，用于指定需要与Data Cache保持一致的存储区域或硬件访问路径。

## 原型定义

```python
PYPTO_DECLARE_ENUM(DcciDst,
    AUTO,             # 根据目标对象自动选择：GM Tensor选CACHELINE_OUT，UB Tile选CACHELINE_UB。
    CACHELINE_OUT,    # 保证Data Cache与GM的一致性，适用于GM Tensor。
    CACHELINE_UB,     # 保证Data Cache与UB的一致性，适用于UB Tile。
    CACHELINE_ALL,    # 与CACHELINE_OUT效果一致，适用于GM Tensor。
    CACHELINE_ATOMIC  # 在硬件原子缓存路径中保证Data Cache与GM的一致性。
)
```

显式指定枚举值时，必须与目标对象的存储区域和硬件访问路径匹配，具体约束请参见[dcci](../cache_control/dcci.md)。
