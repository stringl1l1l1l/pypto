# pypto_pro.language.AccToVecMode

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

双目标控制的枚举，是[pypto_pro.language.move](../memory_data_movement/move.md)的重要属性，用于L0C Buffer->UB数据搬运场景。

## 原型定义

```python
PYPTO_DECLARE_ENUM(AccToVecMode,
     SingleModeVec0,
     SingleModeVec1,
     DualModeSplitM,
     DualModeSplitN
)
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| SingleModeVec0 | 单目标模式，将整个矩阵写入Vec0的目标UB。 |
| SingleModeVec1 | 单目标模式，将整个矩阵写入Vec1的目标UB。 |
| DualModeSplitM | 双目标模式，按M维度拆分，M/2*N个元素写入每个UB。<br>尾块场景下，框架会**自动**在搬运之前对valid_M（L0C Buffer M轴方向的尾块大小）向上对齐到2的倍数获得aligned_M，用户只需要了解UB侧的切分策略：Vec0（sub_id=0）得到前aligned_M / 2行，Vec1（sub_id=1）得到剩余valid_M - aligned_M / 2行。详见[调用示例](#dualmodesplitm模式下的尾块场景)。**注意**：当valid_M为1时，仅切分给Vec0。 |
| DualModeSplitN | 双目标模式，按N维度拆分，M*N/2个元素写入每个UB。<br>尾块场景下，框架会**自动**在搬运之前对valid_N（L0C Buffer N轴方向的尾块大小）向上对齐到32的倍数获得aligned_N，用户只需要了解UB侧的切分策略：Vec0（sub_id=0）得到前aligned_N / 2列，Vec1（sub_id=1）得到剩余valid_N - aligned_N / 2列。详见[调用示例](#dualmodesplitn模式下的尾块场景)。**注意**：当valid_N不超过16时，仅切分给Vec0。 |

## 调用示例

### DualModeSplitM模式下的尾块场景

```python
import pypto_pro.language as pl

# cube section: 用户只需设置实际 valid_M，无需手动对齐
pl.set_validshape(ac, [valid_M, N])       # valid_M=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # 框架自动对齐，aligned_M=34

# vector section: 用户需要自行计算 Vec0/Vec1 中 UB 实际大小
v0 = (valid_M + 1) // 2 * 2 // 2          # v0 = 17
v1 = valid_M - v0                         # v1 = 16
if sub_id == 0:
    pl.set_validshape(vec, [v0, N])
else:
    pl.set_validshape(vec, [v1, N])
```

### DualModeSplitN模式下的尾块场景

```python
import pypto_pro.language as pl

# cube section: 用户只需设置实际 valid_N，无需手动对齐
pl.set_validshape(ac, [TILE, valid_N])     # valid_N=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)  # 框架自动对齐，aligned_N=64

# vector section: V 侧用户自行计算 v0/v1
v0 = (valid_N + 31) // 32 * 32 // 2        # v0 = 32
v1 = valid_N - v0                          # v1 = 1
if sub_id == 0:
    pl.set_validshape(vec, [TILE, v0])
else:
    pl.set_validshape(vec, [TILE, v1])
```
