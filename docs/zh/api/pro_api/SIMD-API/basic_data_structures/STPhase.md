# pypto_pro.language.STPhase

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

矩阵计算结果分阶段写回GM时使用的阶段枚举。用于store和store_tile接口，与矩阵计算接口的[pypto_pro.language.AccPhase](AccPhase.md)配合，保证L0C Buffer中的结果在计算完成后写回，并在最终写回后允许后续计算复用该存储空间。

对应的matmul、matmul_acc、matmul_mx或matmul_mx_acc配置AccPhase时，store或store_tile也必须配置STPhase。矩阵计算的最后一次写操作须使用AccPhase.Final，计算结果的最后一次写回须使用STPhase.Final。对同一计算结果执行多次写回时，最后一次之前使用STPhase.Partial。STPhase不能与Tile类型的scale同时使用。

## 原型定义

```python
PYPTO_DECLARE_ENUM(STPhase,
    Unspecified,  # 不启用分阶段写回，适用于对应矩阵计算未配置AccPhase的场景
    Partial,      # 当前不是对该L0C Buffer计算结果的最后一次写回
    Final         # 当前是对该L0C Buffer计算结果的最后一次写回
)
```
