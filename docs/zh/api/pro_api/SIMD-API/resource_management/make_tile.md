# pypto_pro.language.make_tile

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

按[TileType](../basic_data_structures/TileType.md)在指定Buffer的地址上创建一个Tile，是Kernel中创建Tile的核心接口。Tile的形状、数据类型、Buffer和排布由TileType描述，make_tile将其绑定到具体地址。

如果需要多块同规格Tile构成ping-pong双缓冲并自动管理互斥，使用[pypto_pro.language.make_tile_group](make_tile_group.md)。

下图以UB为例，展示了make_tile如何将Tile绑定到片上地址：addr指定起始地址，地址范围大小由TileType的shape和dtype推导。

![make_tile创建Tile并绑定片上地址](../../figures/make_tile_allocation.jpg "make_tile创建Tile并绑定片上地址")

## 函数原型

```python
pypto_pro.language.make_tile(
    tile_type: TileType,
    *,
    addr: int,
) -> Tile
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile_type | 输入 | Tile类型描述，[TileType](../basic_data_structures/TileType.md)类型，是唯一允许的位置参数。其shape、dtype、target_memory和layout共同决定Tile的存储大小和可用接口。 |
| addr | 输入 | Tile地址，int类型，表示Tile在tile_type所指定Buffer内的字节偏移，必须以关键字形式传入并在编译期确定。UB和L1 Buffer要求32字节对齐，L0A和L0B Buffer要求512字节对齐，L0C Buffer要求64字节对齐。 |

## 约束说明

地址范围的字节数自动由TileType的shape元素数乘以dtype字节数推导，不接受size参数。shape各维必须为编译期正整数；valid_shape不改变分配范围。小于8位的数据类型按每个元素至少1字节保留空间。

多个Tile的地址范围不得发生非预期重叠；需要有意复用同一块Buffer时，调用方必须自行保证访存时序正确。

## 返回值说明

返回绑定到指定片上地址的Tile。

## 调用示例

### 在UB中创建Tile并完成逐元素加法

```python
import pypto_pro.language as pl


@pl.jit()
def make_tile_add_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    # 地址范围大小由tt推导为64 * 128 * 2 = 16384字节
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_b = pl.make_tile(tt, addr=0x4000)
    tile_out = pl.make_tile(tt, addr=0x8000)

    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```

Tile的形状、数据类型、Buffer、排布和尾块配置见[pypto_pro.language.TileType](../basic_data_structures/TileType.md)。
