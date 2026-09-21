# pypto_pro.language.add

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

两个操作数对应位置逐元素做加法。支持Tile-Tile和Tile-Scalar两种模式，其中Scalar为标量；同时支持原地计算。

- **Tile-Tile模式**：对lhs和rhs对应位置的元素求和，将结果写入out。
- **Tile-Scalar模式**：将lhs中的每个元素与rhs标量相加，将结果写入out。

## 函数原型

```python
pypto_pro.language.add(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素加法的结果。数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。可与lhs或Tile类型的rhs为同一Tile，实现原地计算。 |
| lhs | 输入 | 左操作数，Tile类型。数据类型与out一致。 |
| rhs | 输入 | 右操作数，Tile或Scalar类型。传入Tile时执行Tile-Tile计算，数据类型与out一致，且shape与out、lhs一致；传入Scalar时执行Tile-Scalar计算。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### Tile-Tile模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下。

<!-- pypto-doc-output:add:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[11 11.75 12.5 13.25 14 14.75 15.5 16.25 ...], [59 59.75 60.5 61.25 62 62.75 63.5 64.25 ...], [107 107.75 108.5 109.25 110 110.75 111.5 112.25 ...], [155 155.75 156.5 157.25 158 158.75 159.5 160.25 ...], ...]
```
<!-- pypto-doc-output:add:end -->

### Tile-Scalar模式

```python
# Tile每个元素加上Scalar值。
pl.add(out, lhs, 1.0)
```
