# pypto_pro.language.expands

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

将Tile填充为指定标量值。常用于初始化负无穷Tile或零Tile。

## 函数原型

```python
pypto_pro.language.expands(
    out: Tile,
    scalar: Scalar,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，全部元素被填充为scalar值。<br>数据类型支持：DT_UINT8、DT_INT8、DT_UINT16、DT_INT16、DT_UINT32、DT_INT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16、DT_FP32。<br>位于UB或L1 Buffer。 |
| scalar | 输入 | 填充值。为整型或浮点型常量，或运行时整型或浮点型标量表达式，类型须与out元素类型兼容。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl

K_VALUE = 2.0


@pl.jit(auto_mutex=True)
def expands_kernel(dummy: pl.Tensor[[64, 64], pl.DT_FP32],
                   out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        cur_out = tile_out.current()
        pl.expands(cur_out, K_VALUE)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:expands:start -->
```bash
输入数据K_VALUE：2
输出数据out：[[2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:expands:end -->

### 初始化负无穷Tile

```python
# 初始化负无穷 Tile（因果掩码）
pl.expands(neg_inf_vec, NEG_INF)
```

### 初始化零Tile

```python
# 初始化零 Tile
pl.expands(score_u16_row, 0)
```
