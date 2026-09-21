# pypto_pro.language.dequant

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

把INT8或INT16源Tile反量化为FP32。scale[i,0]和offset[i,0]按行广播：

$$
out_{i,j}=\left(\operatorname{FP32}(src_{i,j})-offset_{i,0}\right)\times scale_{i,0}
$$

计算顺序为：整数源数据扩展并转换为FP32、减去offset、乘以scale。若与[quant](quant.md)配套使用，量化阶段传入的系数通常为本接口scale的倒数。

## 函数原型

```python
pypto_pro.language.dequant(
    out: Tile,
    src: Tile,
    scale: Tile,
    offset: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存储空间为UB，数据类型为DT_FP32，layout必须为ND；shape和valid_shape须与src一致。 |
| src | 输入 | 源操作数，Tile类型，存储空间为UB，数据类型为DT_INT8或DT_INT16，layout必须为ND；shape和valid_shape须与out一致。 |
| scale | 输入 | 反量化系数，Tile类型，存储空间为UB，数据类型为DT_FP32，layout为ND或DN。若out.valid_shape=[M,N]，则scale.valid_shape须为[M,1]，shape的行数不得小于M且列数必须为1；有效行数须与输出有效行数一致，物理行数须满足32字节对齐，DT_FP32场景即为8的倍数。第i行的scale[i,0]广播到输出第i行全部有效列。 |
| offset | 输入 | 量化零点，Tile类型，存储空间为UB，数据类型为DT_FP32，layout为ND或DN，shape和valid_shape均须与scale一致；有效行数须与输出有效行数一致，物理行数须满足32字节对齐，DT_FP32场景即为8的倍数。第i行的offset[i,0]广播到对应数据行。本参数不能省略；对称量化时传入全零Tile。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dequant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_INT8],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    offset: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x4000, mutex_ids=[1])
    tile_offset = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                     addrs=0x5000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x6000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_offset = tile_offset.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.load(cur_offset, offset, [0, 0])
        pl.dequant(cur_out, cur_src, cur_scale, cur_offset)
        pl.store(out, cur_out, [0, 0])
```
