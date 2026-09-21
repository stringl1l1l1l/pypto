# pypto_pro.language.xor

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

对源操作数lhs和rhs对应位置的元素执行按位异或，将结果写入目的操作数out。

## 函数原型

```python
pypto_pro.language.xor(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    tmp: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素按位异或的结果。数据类型与lhs、rhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32或DT_UINT32。shape与lhs、rhs一致。 |
| lhs | 输入 | 左操作数，Tile类型。数据类型与out一致。 |
| rhs | 输入 | 右操作数，Tile类型。采用row-major布局，valid_shape与out一致，shape与out、lhs一致。数据类型与out一致。 |
| tmp | 输入 | 兼容性参数，Tile类型。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def xor_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    b: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.xor(cur_out, cur_a, cur_b, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下。

<!-- pypto-doc-output:xor:start -->
```bash
输入数据a：[[2 3 4 5 6 7 8 9 ...], [66 67 68 69 70 71 72 73 ...], [130 131 132 133 134 135 136 137 ...], [194 195 196 197 198 199 200 201 ...], ...]
输入数据b：[[1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], ...]
输出数据out：[[3 1 7 1 3 1 15 1 ...], [67 65 71 65 67 65 79 65 ...], [131 129 135 129 131 129 143 129 ...], [195 193 199 193 195 193 207 193 ...], ...]
```
<!-- pypto-doc-output:xor:end -->
