# pypto_pro.language.quant

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

把FP32源Tile量化为INT8或UINT8。scale是量化系数（常见量化定义中真实scale的倒数），scale[i, 0]和offset[i, 0]按行广播：

$$
out_{i,j}=\begin{cases}
\operatorname{clamp}_{[-128,127]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}\right)\right), & mode=\mathrm{SYM} \\
\operatorname{clamp}_{[0,255]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}+offset_{i,0}\right)\right), & mode=\mathrm{ASYM}
\end{cases}
$$

非对称模式先在FP32中完成乘法和加法，再对合并后的结果执行一次舍入；不是先舍入乘积再添加offset。

## 函数原型

```python
pypto_pro.language.quant(
    out: Tile,
    src: Tile,
    scale: Tile,
    *,
    mode: QuantMode = pypto_pro.language.QuantMode.SYM,
    offset: Optional[Tile] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存储空间为UB，layout必须为ND。mode=SYM时数据类型为DT_INT8，mode=ASYM时为DT_UINT8；shape和valid_shape须与src一致。 |
| src | 输入 | 源操作数，Tile类型，存储空间为UB，数据类型为DT_FP32，layout必须为ND；shape和valid_shape须与out一致。 |
| scale | 输入 | 量化系数，Tile类型，存储空间为UB，数据类型为DT_FP32，layout为ND或DN，取值须为有限正数。若src.valid_shape=[M,N]，则scale.valid_shape须为[M,1]，shape的行数不得小于M且列数必须为1；有效行数须与输出有效行数一致，物理行数须满足32字节对齐，DT_FP32场景即为8的倍数。第i行的scale[i,0]广播到src第i行全部有效列。该值直接与src相乘，因此若使用常见公式$q=round(x/s)$，此处应传入$1/s$。 |
| mode | 输入 | 可选，编译期[pypto_pro.language.QuantMode](../basic_data_structures/QuantMode.md)枚举值，不能使用运行时Scalar或Tensor动态指定；可取SYM（默认）或ASYM。模式同时决定输出dtype和是否需要offset。 |
| offset | 输入 | 量化零点，Tile类型。mode=ASYM时必填，mode=SYM时建议省略；存储空间为UB，数据类型为DT_FP32，layout为ND或DN，shape和valid_shape均须与scale一致；有效行数须与输出有效行数一致，物理行数须满足32字节对齐，DT_FP32场景即为8的倍数。第i行的offset[i,0]广播到对应数据行。 |

## 约束说明

- 计算结果采用就近舍入（中间值取偶），并饱和至目标数据类型的取值范围。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def quant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_INT8],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0xA000, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.quant(cur_out, cur_src, cur_scale, mode=pl.QuantMode.SYM)
        pl.store(out, cur_out, [0, 0])
```
