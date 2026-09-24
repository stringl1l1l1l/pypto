# pypto_pro.language.select

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

按掩码逐元素选择数据，掩码为真时取lhs，为假时取rhs。rhs可以是Tile或Scalar。该接口通常与pypto_pro.language.eq、ne、lt、le、gt、ge配合使用。

## 函数原型

```python
pypto_pro.language.select(
    out: Tile,
    mask: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
    tmp: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存储空间为UB，形状必须与输入数据Tile一致。Tile-Tile和Tile-Scalar模式均支持8、16、32、64位整型、DT_FP16、DT_BF16和DT_FP32。 |
| mask | 输入 | 源操作数（掩码），Tile类型，存储空间为UB，采用按位压缩格式，掩码Tile的数据类型不做限制，通常使用DT_UINT8类型，须由比较接口生成。 |
| lhs | 输入 | 源操作数（掩码为真时选取的数据），Tile类型，存储空间为UB，数据类型和形状必须与out一致。 |
| rhs | 输入 | 源操作数（掩码为假时选取的数据），Tile或Scalar类型，也支持可转换为Scalar的Python int或float常量。传入Tile时，存储空间为UB，数据类型和形状必须与out一致；传入Scalar或Python常量时，数据类型必须与out的元素类型兼容。 |
| tmp | 输入 | 兼容性参数，Tile类型。 |

## 约束说明

- out、mask、lhs以及Tile类型的rhs必须位于UB。out、lhs以及Tile类型的rhs必须采用行主序排布。
- out、lhs以及Tile类型的rhs必须具有相同的形状、有效形状和数据类型。

## 返回值说明

无。

## 调用示例

### 根据比较结果在两个Tile间选择

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scalar_gt_select_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    b: pl.Tensor[[64, 128], pl.DT_FP32],
    mask_in: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt32 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt32, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt32, addrs=0x8000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt32, addrs=0x10000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt32, addrs=0x18000, mutex_ids=[3])
    mask_fp16_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000, mutex_ids=[4])
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x24000, mutex_ids=[5])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_fp16 = mask_fp16_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.load(mask_fp16, mask_in, [0, 0])
        # mask_fp16 > 0，生成按位压缩的掩码mask_vec
        pl.gt(mask_vec, mask_fp16, 0.0)
        # 谓词为真取 lhs(=a)，否则取 rhs(=b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])
```

### 运行结果

<!-- pypto-doc-output:select:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据b：[[8 7.875 7.75 7.625 7.5 7.375 7.25 7.125 ...], [-8 -8.125 -8.25 -8.375 -8.5 -8.625 -8.75 -8.875 ...], [-24 -24.125 -24.25 -24.375 -24.5 -24.625 -24.75 -24.875 ...], [-40 -40.125 -40.25 -40.375 -40.5 -40.625 -40.75 -40.875 ...], ...]
输入数据mask：[[1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], ...]
输出数据out：[[1 7.875 1.5 7.625 2 7.375 2.5 7.125 ...], [33 -8.125 33.5 -8.375 34 -8.625 34.5 -8.875 ...], [65 -24.125 65.5 -24.375 66 -24.625 66.5 -24.875 ...], [97 -40.125 97.5 -40.375 98 -40.625 98.5 -40.875 ...], ...]
```
<!-- pypto-doc-output:select:end -->
