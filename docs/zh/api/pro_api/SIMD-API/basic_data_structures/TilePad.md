# pypto_pro.language.TilePad

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

Tile边界不足时的填充方式枚举，用于尾块/非满块场景。

当Tile的有效数据区域小于其shape时（如动态维度的尾块），超出有效区域的部分需要按指定模式填充。

## 原型定义

```python
PYPTO_DECLARE_ENUM(
    TilePad,
    null,
    zero,
    max,
    min
)
```

## 参数说明

| 参数值 | 说明 |
|---|---|
| null | 不填充，默认值，适用于不需要处理无效区域的场景。 |
| zero | 补0，典型用于卷积padding和零初始化。 |
| max | 补对应数据类型的最大值，典型用于取最小值操作的无效区域。 |
| min | 补对应数据类型的最小值，典型用于取最大值操作的无效区域。 |

## 约束说明

无。

## 调用示例

### Tile填充模式

```python
import pypto_pro.language as pl
# 不填充（默认）
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec)

# 补零
tt_pad = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
```
