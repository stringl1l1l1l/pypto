# pypto_pro.language.SyncCoreType

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

SyncCoreType是指定sync_all参与核类型的枚举。

## 原型定义

```python
PYPTO_DECLARE_ENUM(SyncCoreType,
    AIV_ONLY,  # 仅同步参与执行的AIV核
    AIC_ONLY,  # 仅同步参与执行的AIC核
    MIX        # 同步参与执行的AIC核和AIV核
)
```
