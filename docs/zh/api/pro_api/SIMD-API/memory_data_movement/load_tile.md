# pypto_pro.language.load_tile

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

将GM Tensor中的数据搬入L1 Buffer或UB中的Tile。与[pypto_pro.language.load](load.md)不同，tile_offsets使用Tile块编号指定搬运位置：由order选中的Tensor维度按Tile块编号寻址，接口将对应编号乘以dst_tile的shape中对应维度的大小，换算为绝对元素坐标；未被order选中的维度仍按绝对元素坐标寻址。

例如，对于shape=[64, 128]的二维Tile，不设置order时，tile_offsets=[2, 2]对应的绝对元素坐标为[128, 256]，等价于调用pypto_pro.language.load时传入offsets=[128, 256]。

![load_tile按块索引从GM搬入Tile](../../figures/load_tile_block_offset.jpg "load_tile按块索引从GM搬入Tile")

## 函数原型

```python
pypto_pro.language.load_tile(
    dst_tile: Tile,
    src_tensor: Tensor,
    tile_offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L1 Buffer或UB，首地址必须按32字节对齐。接口按照该Tile的valid_shape搬运数据；Tile块大小由shape决定，不受valid_shape影响。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tensor | 输入 | 源操作数，Tensor类型，存储空间为GM。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| tile_offsets | 输入 | 表示源Tensor各维度的Tile块编号或绝对元素坐标，List[int或Scalar]类型，长度须与源Tensor的维数相同。<br>- 不支持负数。<br>- order选中的维度按Tile块编号寻址，对应编号乘以dst_tile的shape中对应维度的大小后得到绝对元素坐标。<br>- order未选中的维度按绝对元素坐标寻址，用于固定高维Tensor的其他维度。<br>- 换算后的绝对元素坐标必须位于源Tensor的shape范围内；边界Tile可通过pypto_pro.language.set_validshape设置有效形状，保证有效搬运范围不越过源Tensor边界。 |
| order | 输入 | 可选，维度映射，List[int]类型。列表中的第i项表示dst_tile第i维对应src_tensor的维度。<br>- 升序表示不转置，例如order=[0, 1]。<br>- 降序表示转置，例如order=[1, 0]。<br>- 不设置时，dst_tile默认对应src_tensor的最后两个维度，且不转置。 |

## 约束说明

- 数据类型及分形约束：

  pypto_pro.language.load_tile与pypto_pro.language.load支持的数据类型和分形组合相同，详见[pypto_pro.language.load](load.md#约束说明)。

- 偏移及维度映射约束：

  - tile_offsets的长度必须与src_tensor的维数相同，各项必须为非负整数或运行时整数表达式。
  - order必须是长度为2的编译期整数列表，两个维度索引必须互不重复，且位于src_tensor的维度范围内。未被order选中的维度由tile_offsets中的对应值固定为一个下标。
  - 换算后的绝对元素坐标必须位于src_tensor的shape范围内。边界Tile的有效搬运范围不得越过src_tensor边界，可通过pypto_pro.language.set_validshape设置dst_tile的有效形状。

- NZ搬运约束：

  - src_tensor声明为NZ时，dst_tile必须为NZ，且只支持从src_tensor的最后两个维度正序搬运，不支持通过降序order转置。NZ的物理排布和Tensor shape约束详见[TensorLayout](../basic_data_structures/TensorLayout.md)。
  - tile_offsets换算后的M方向偏移必须按16对齐，N方向偏移必须按src_tensor数据类型对应的C0对齐；dst_tile的shape和valid_shape需要满足相同的对齐要求。

- MX矩阵乘量化系数搬运约束：

  - 仅支持将DT_FP8E8M0的src_tensor搬入fractal为32、布局为ZZ或NN的L1 Buffer Tile，并作为pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc的量化系数使用。
  - src_tensor的维数必须大于等于3，最后一维为长度等于2的物理phase轴。order必须显式选择量化系数矩阵对应的两个维度，不能选择phase轴；tile_offsets中phase轴对应的偏移必须为0。

- Tile地址复用约束：

  pypto_pro.language.load_tile与pypto_pro.language.load的地址复用规则相同。连续两次搬运写入同一片上地址时，按照[pypto_pro.language.load](load.md#约束说明)中的Tile地址复用约束进行同步。

## 返回值说明

无。

## 调用示例

### 按Tile块编号搬运

输入Tensor由4个[64, 64]数据块组成。循环变量tile_index依次取0～3，load_tile自动将其换算为第0维的绝对元素偏移0、64、128和192。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_kernel(
    x: pl.Tensor[[256, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tiles = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    out_tiles = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])

    with pl.section_vector():
        for tile_index in pl.range(0, 4, 1):
            current_x = x_tiles.next()
            current_out = out_tiles.next()
            pl.load_tile(current_x, x, [tile_index, 0])
            pl.add(current_out, current_x, current_x)
            pl.store_tile(out, current_out, [tile_index, 0])
```

### 从高维Tensor中选择维度搬运

以下示例中，输入Tensor的维度依次为B、S、N、D。order=[1, 3]表示Tile的两个维度分别对应S、D，因此tile_offsets中的S、D按Tile块编号解释，B、N仍按绝对下标解释。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_high_dim_kernel(
    x: pl.Tensor[[2, 128, 4, 64], pl.DT_FP16],
    out: pl.Tensor[[2, 128, 4, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])

    with pl.section_vector():
        current_x = x_tile.current()
        current_out = out_tile.current()
        # 固定B=1、N=2；S方向块编号为1，对应绝对元素偏移64。
        pl.load_tile(current_x, x, [1, 1, 2, 0], order=[1, 3])
        pl.add(current_out, current_x, current_x)
        pl.store_tile(out, current_out, [1, 1, 2, 0], order=[1, 3])
```

### 转置搬运

转置场景同样通过降序order指定。以下示例使用shape=[64, 64]的方形Tile，tile_offsets=[1, 0]先换算为绝对元素坐标[64, 0]，再从普通ND GM Tensor转置搬入ZN Tile。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_transpose_kernel(x: pl.Tensor[[128, 128], pl.DT_FP16]):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
    )
    x_l1 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_cube():
        pl.load_tile(x_l1.current(), x, [1, 0], order=[1, 0])
```
