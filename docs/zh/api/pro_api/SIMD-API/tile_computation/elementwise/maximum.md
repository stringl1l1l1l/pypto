# pypto_pro.language.maximum

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

maximum同时支持逐元素取较大值和按维度取最大值归约。是否传入dim决定调用模式。

- **Tile-Tile模式**：比较lhs和rhs对应位置的元素，将较大值写入out。
- **Tile-Scalar模式**：比较lhs中的每个元素与rhs标量，将较大值写入out。
- **归约模式**：对lhs沿dim指定的维度取最大值，将结果写入out。

## 函数原型

```python
pypto_pro.language.maximum(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
    *,
    dim: Optional[int] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素计算结果或归约结果。逐元素模式下数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32；归约模式下数据类型与lhs一致，dim=0时shape为[行数, 1]，dim=1时shape为[1, 列数]。 |
| lhs | 输入 | Tile类型。逐元素模式下为左操作数；归约模式下为源操作数，必须为二维Tile，dim=0时支持DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_FP16、DT_FP32、DT_INT64和DT_UINT64，dim=1时还支持DT_UINT16、DT_UINT32和DT_BF16。 |
| rhs | 输入 | Tile或Scalar类型。逐元素模式下为右操作数，Tile-Tile时数据类型与out一致且shape与out、lhs一致；归约模式下为临时Tile。 |
| dim | 输入 | 可选，归约维度。未传入时执行逐元素计算；传入0时沿最后一维归约；传入1时沿第一维归约。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### Tile-Tile模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def maximum_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.maximum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下。

<!-- pypto-doc-output:maximum:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
```
<!-- pypto-doc-output:maximum:end -->

### Tile-Scalar模式

```python
# Tile每个元素与Scalar值取较大值。
pl.maximum(out, lhs, 0.0)
```

### 归约模式

```python
pl.maximum(row_out, src, tmp, dim=0)  # row_out shape：[行数, 1]
pl.maximum(col_out, src, tmp, dim=1)  # col_out shape：[1, 列数]
```

#### dim = 0实测结果

```python
@pl.jit(auto_mutex=True)
def row_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_max:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[23.75], [55.75], [87.75], [119.75], ...]
```
<!-- pypto-doc-output:row_max:end -->

#### dim = 1实测结果

```python
@pl.jit(auto_mutex=True)
def col_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_max:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[2.008000e+03 2.008250e+03 2.008500e+03 2.008750e+03 2.009000e+03 2.009250e+03 2.009500e+03 2.009750e+03 ...]]
```
<!-- pypto-doc-output:col_max:end -->
