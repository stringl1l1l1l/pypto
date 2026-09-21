# pypto_pro.language.QuantMode

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

量化模式枚举，用于指定quant接口执行对称量化或非对称量化。

## 原型定义

```python
PYPTO_DECLARE_ENUM(QuantMode,
    SYM,   # 对称量化，输出DT_INT8，不使用offset参数。
    ASYM   # 非对称量化，输出DT_UINT8，须提供DT_FP32类型的offset Tile。
)
```

计算公式、参数形状和数据类型约束，请参见[quant](../quantization/quant.md)。
