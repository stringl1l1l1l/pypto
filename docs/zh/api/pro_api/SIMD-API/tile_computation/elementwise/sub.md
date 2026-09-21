# pypto_pro.language.sub

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

两个操作数对应位置逐元素做减法。支持Tile-Tile和Tile-Scalar两种模式，其中Scalar为标量；同时支持原地计算。

- **Tile-Tile模式**：将lhs对应位置的元素减去rhs中的元素，将结果写入out。
- **Tile-Scalar模式**：将lhs中的每个元素减去rhs标量，将结果写入out。

## 函数原型

```python
pypto_pro.language.sub(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素减法的结果。数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。可与lhs或Tile类型的rhs为同一Tile，实现原地计算。 |
| lhs | 输入 | 左操作数（被减数），Tile类型。数据类型与out一致。 |
| rhs | 输入 | 右操作数（减数），Tile或Scalar类型。传入Tile时执行Tile-Tile计算，数据类型与out一致，且shape与out、lhs一致；传入Scalar时执行Tile-Scalar计算。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### Tile-Tile模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sub_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.sub(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下。

<!-- pypto-doc-output:sub:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[-9 -9.25 -9.5 -9.75 -10 -10.25 -10.5 -10.75 ...], [-25 -25.25 -25.5 -25.75 -26 -26.25 -26.5 -26.75 ...], [-41 -41.25 -41.5 -41.75 -42 -42.25 -42.5 -42.75 ...], [-57 -57.25 -57.5 -57.75 -58 -58.25 -58.5 -58.75 ...], ...]
```
<!-- pypto-doc-output:sub:end -->

### Tile-Scalar模式

```python
# Tile每个元素减去Scalar值。
pl.sub(out, lhs, 1.0)
```
