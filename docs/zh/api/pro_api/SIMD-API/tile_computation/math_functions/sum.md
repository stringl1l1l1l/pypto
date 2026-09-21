# pypto_pro.language.sum

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

沿指定维度对源Tile求和。dim=0时沿行方向对每行求和；dim=1沿列方向对每列求和。

## 函数原型

```python
pypto_pro.language.sum(
    out: Tile,
    src: Tile,
    tmp: Tile,
    *,
    dim: int = 0,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放归约结果，支持的数据类型详见[约束说明](#约束说明)。<br>如果src的shape为[M, N]。dim=0时shape为[M, 1]；dim=1时shape为[1, N]。 |
| src | 输入 | 源操作数，Tile类型，支持的数据类型详见[约束说明](#约束说明)。 |
| tmp | 输入 | 临时存储，Tile类型，兼容性参数。<br>dim=0降低到TROWSUM时不读写该参数；dim=1降低到TCOLSUM，默认非binary路径不使用该参数。 |
| dim | 输入 | 归约维度，0表示沿行方向做归约，1表示沿列方向做归约。 |

## 约束说明

- 数据类型：dim取值不同时，src和out支持的数据类型支持范围不同，具体如下。

  | 参数 | dim=0 | dim=1 |
  |---|---|---|
  | src | DT_INT16、DT_INT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32 | DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16、DT_FP32 |
  | out | 与src保持一致 | 与src保持一致 |

- FP16精度：FP16归约会受到输入量化、有限精度累加及输出舍入的影响。输入规模较大或数值较大时，设备计算结果可能与高精度参考结果存在差异。归约指令采用的累加顺序也可能影响最终结果。

## 返回值说明

无。

## 调用示例

### dim = 0

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.sum(cur_out, cur_a, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[1.008000e+03], [5.104000e+03], [9.200000e+03], [1.329600e+04], ...]
```
<!-- pypto-doc-output:row_sum:end -->

### dim = 1

```python
@pl.jit(auto_mutex=True)
def col_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[6.400000e+04 6.401600e+04 6.403200e+04 6.404800e+04 6.406400e+04 6.408000e+04 6.409600e+04 6.411200e+04 ...]]
```
<!-- pypto-doc-output:col_sum:end -->

### FP16输入下的精度差异

FP16归约会受到输入量化、有限精度累加及输出舍入的影响。输入规模较大或数值较大时，设备计算结果可能与高精度参考结果存在差异。归约指令采用的累加顺序也可能影响最终结果。

以下为sum(..., dim=1)使用FP16输入时的设备实测结果：

<!-- pypto-doc-output:col_reduce_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[6.390400e+04 6.390400e+04 6.390400e+04 6.416000e+04 6.416000e+04 6.416000e+04 6.419200e+04 6.419200e+04 ...]]
```
<!-- pypto-doc-output:col_reduce_sum:end -->
