# pypto_pro.language.abs

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

计算源操作数src中每个元素的绝对值，将结果写入目的操作数out。负值取相反数，正值和零保持不变。

## 函数原型

```python
pypto_pro.language.abs(
    out: Tile,
    src: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素取绝对值的结果。数据类型与src一致，支持DT_FP16、DT_FP32、DT_INT8、DT_INT16或DT_INT32。shape和valid_shape与src一致。可与src为同一Tile，实现原地计算。 |
| src | 输入 | 源操作数，Tile类型。数据类型与out一致。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def abs_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.abs(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
