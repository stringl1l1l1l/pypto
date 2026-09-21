# pypto_pro.language.TileType

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

描述一块Tile的规格，包括形状、数据类型、所在内存空间和排布方式等，配合[pypto_pro.language.make_tile](../resource_management/make_tile.md)或[pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md)分配实际缓冲区。

TileType本身不分配内存，只是一个规格描述符。实际缓冲区通过make_tile（单块）或make_tile_group（多块轮转）创建。

## 函数原型

```python
pypto_pro.language.TileType(
    shape: Sequence[int],
    dtype: DataType,
    target_memory: MemorySpace = pypto_pro.language.MemorySpace.Vec,
    valid_shape: Optional[Sequence[int]] = None,
    layout: Optional[TensorLayout] = None,
    fractal: Optional[int] = None,
    pad: Optional[Union[TilePad, int]] = None,
    compact: Optional[int] = None,
) -> TileType
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| shape | 输入 | Tile各维大小，如[64, 128]。长度为2的编译期常量整数列表，各维大小须为正整数；仅支持二维Tile。对齐及分形布局约束由使用该TileType的具体API检查。 |
| dtype | 输入 | 元素的数据类型，[pypto_pro.language.DataType](DataType.md)类型。 |
| target_memory | 输入 | 目标内存空间，[pypto_pro.language.MemorySpace](MemorySpace.md)枚举值，默认为UB。 |
| valid_shape | 输入 | 可选，有效形状（处理尾块/非满块场景）。编译期常量整数列表或None（默认）。<br>- 具体整数（如[32, 64]）：编译期确定有效形状。<br>- None（默认）：等同于[-1, -1]（动态模式）。<br>- [-1, -1]：运行时动态设置有效形状，配合pypto_pro.language.set_validshape使用。 |
| layout | 输入 | 可选，排布方式，[pypto_pro.language.TensorLayout](TensorLayout.md)枚举值或None（默认）。不指定时按内存空间取默认值（详见[约束说明](#约束说明)）。L1 Buffer未指定layout时使用NZ，也可以显式指定ZN；数据类型为DT_UINT64或DT_INT64时，还可以显式指定ND。BiasTable Buffer默认且仅支持ND；L0A_MX Buffer仅支持ZZ，L0B_MX Buffer仅支持NN；UB的可用布局由具体Tile API约束。 |
| fractal | 输入 | 可选，分形大小，整数或None（默认）。L0C Buffer中的FP32/INT32 Tile在未指定时自动设为1024；L1 Buffer中layout为ZZ或NN的DT_FP8E8M0 Tile及L0A_MX Buffer/L0B_MX Buffer中的Tile默认且仅支持fractal=32；其他取值要求由具体Tile API决定。 |
| pad | 输入 | 可选，填充模式，支持[pypto_pro.language.TilePad](TilePad.md)类型或int类型。当为int类型时取值范围为[0, 3]：<br>- 0：不填充。<br>- 1：补零。<br>- 2：补最大值。<br>- 3：补最小值。<br>非法值报ValueError，非法类型报TypeError。 |
| compact | 输入 | 可选，Tile缓冲区的紧凑布局模式，描述Tile在搬运、重排和矩阵计算路径中的布局解释方式，不改变数据类型，也不代替valid_shape对实际有效区域的描述。取值如下：<br>- **0或None**：不启用紧凑模式。此时L1 Buffer→L0A Buffer/L0B Buffer的[move](../memory_data_movement/move.md)完全按物理shape搬运，set_validshape在该搬运路径上不生效。<br>- **1**：使用普通紧凑模式。数据在valid_shape向上对齐到分形粒度的有效空间内连续排布，Tile声明的其余空间空闲在尾部，详见[普通紧凑模式的数据排布](#普通紧凑模式的数据排布)。通常用于尾块场景，与set_validshape搭配使用。在L1 Buffer上配置与否对结果无影响。与phase搭配使用时，如果存在尾块，L0C Buffer必须配置compact=1，否则可能会卡死。compact不会填充无效区域，需要填充时通过pad参数或fillpad补齐。compact不改变缓冲区的分配大小，只改变数据在缓冲区内的排布，因此把多个Tile拼成一个更宽的操作数时需特别注意地址偏移，详见[多Tile拼接的尾块处理](#多tile拼接的尾块处理)。<br>- **2**：使用RowPlusOne紧凑模式。仅在UB Tile中配置，用于避免以该Tile为源作搬运时的bank冲突，详见[RowPlusOne紧凑模式的数据排布](#rowplusone紧凑模式的数据排布)。NZ格式下每个分形列多预留一行物理空间，仅作占位不参与计算，因此申请Tile的物理shape时须包含多出来的这一行，数据使用时通过set_validshape配置实际的有效行数；ZN格式同理，多预留的是一列。 |

## 约束说明

### layout默认值说明

layout不指定时，按内存空间自动取默认值：

| 内存空间 | 默认layout |
|---|---|
| UB | 无约束 |
| L1 Buffer | NZ |
| L0A Buffer | NZ |
| L0B Buffer | ZN |
| L0C Buffer | NZ |
| BiasTable Buffer | ND |
| Fixpipe Buffer | ND |
| L0A_MX Buffer | ZZ |
| L0B_MX Buffer | NN |


### 普通紧凑模式的数据排布

```text
  举例：L0A Buffer中Tile的物理shape=[64, 64]，数据类型为FP16，valid_shape=[8, 24]，NZ格式，由此可知：
        1、分形粒度D=16行×16列，共512字节（0x200）；
        2、实际搬运数据的有效空间为[ceil16(8), ceil16(24)] = [16, 32]，共2个分形；

  不配置 compact：数据按 shape=[64,64] 排布

     0   16   32   48   64  ← 列
   0 ┌────┬────┬────┬────┐
     │ D1 │ D2 │    │    │   D1、D2 = 两个有效数据分形
  16 ├────┼────┼────┼────┤
     │    │    │    │    │   相邻两列的间距 = 64 行 = 4 个分形 = 0x800
  32 ├────┼────┼────┼────┤   D1 首地址 0x000，D2 首地址 0x800
     │    │    │    │    │
  48 ├────┼────┼────┼────┤
     │    │    │    │    │
  64 └────┴────┴────┴────┘

  配置 compact=1：数据在 [16,32]，有效空间内连续排布

     0   16   32
   0 ┌────┬────┐          相邻两列的间距 = 16 行 = 1 个分形 = 0x200
     │ D1 │ D2 │          D1 首地址 0x000，D2 首地址 0x200
  16 └────┴────┘
     ┌─────────────────────┐
     │ 声明的其余空间，空闲 │   ← 缓冲区仍按 shape 分配
     └─────────────────────┘
```

### RowPlusOne紧凑模式的数据排布

```text
  举例：UB Tile的物理shape=[64, 64]，NZ格式

  配置 compact=1：分形列首尾相接

     ┌────┬────┬────┐
     │ D1 │ D2 │ D3 │   每列 64 行
     └────┴────┴────┘
       ↑    ↑    ↑
     首地址间距 = 64 行，各列起始落在相同的 bank 上


  配置 compact=2：每列多留 1 行

     ┌─────┬─────┬─────┐
     │ D1  │ D2  │ D3  │   每列 64 行数据
     │ ─── │ ─── │ ─── │ ← 每列末尾多出的 1 行（占位）
     └─────┴─────┴─────┘
       ↑     ↑     ↑
     首地址间距 = 65 行，各列起始依次错开一行，落到不同 bank 上
```

## 返回值说明

返回一个Tile类型描述对象。该对象仅保存Tile的shape、dtype、内存空间和布局等属性，不分配片上内存；需传给make_tile或make_tile_group创建实际Tile。

## 调用示例

TileType仅描述Tile规格，实际缓冲区由make_tile或make_tile_group创建。

### 默认UB Tile

```python
import pypto_pro.language as pl

# target_memory 默认取 MemorySpace.Vec
vec_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16)
```

### Cube各内存空间的Tile

```python
import pypto_pro.language as pl

# L0A Buffer中的Tile：Left的默认布局为NZ，也可以显式指定
left_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left, layout=pl.NZ,
)

# L1 Buffer中的转置分形Tile：Mat支持显式指定ZN
mat_zn_type = pl.TileType(
    shape=[128, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
)

# L0C Buffer中的Tile：FP32 Acc未指定fractal时自动取1024
acc_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Acc,
)
```

### 动态尾块Tile

```python
import pypto_pro.language as pl

# valid_shape=[-1, -1]：运行时通过 set_validshape 设置实际有效形状
# compact=1：使动态尾块在 load、move 和 matmul 路径中按有效窗口紧凑解释
tail_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1], compact=1,
)
```

### compact = 1 行方向尾块

```python
import os
import pypto_pro.language as pl
import torch

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
TILE = 128
M_TAIL = 72

@pl.jit(auto_mutex=True)
def matmul_m_tail(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[M_TAIL, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ, valid_shape=[-1, -1],
        ),
        addrs=0x0,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    # L0A Buffer：matmul恒按紧凑布局取数，compact=1使move按同样的跨度写入
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
            layout=pl.NZ, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0,
        mutex_ids=[3],
    )
    # L0C Buffer：matmul恒按紧凑布局写数，compact=1使store按同样的跨度读出
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(al, [M_TAIL, TILE])
        pl.set_validshape(ac, [M_TAIL, TILE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])

if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    torch.manual_seed(42)
    a = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    b = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    out = torch.zeros([M_TAIL, TILE], device=DEVICE, dtype=torch.float32)

    matmul_m_tail(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a[:M_TAIL, :].float(), b.float())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### 多Tile拼接的尾块处理

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def matmul_wide_alias(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[40, 128], pl.DT_FP32],
):
    x_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    y_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    x_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[2, 3],
    )
    # 别名：同一基地址，作为matmul的左操作数
    x_wide = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[[2, 3]],
    )
    y_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        cur_x = x_l1.current()
        cur_y = y_l1.current()
        br = y_l0b.current()
        ac = c_l0c.current()

        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.move(br, cur_y)

        # 两块L0A Buffer中的Tile不设置compact=1和set_validshape，数据全量搬运，确保拼接后的整块L0A Buffer中数据排布连续
        al_1 = x_l0a.next()
        pl.move(al_1, cur_x, [0, 0])
        al_2 = x_l0a.next()
        pl.move(al_2, cur_x, [0, 64])

        al = x_wide.next()
        pl.matmul(ac, al, br)

        # L0C Buffer不设置compact=1，store按matmul的写入格式搬运；set_validshape控制搬运的数据量；
        pl.set_validshape(ac, [40, 128])
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn([64, 128], device=device, dtype=torch.float16)
    y = torch.eye(128, device=device, dtype=torch.float16)
    out = torch.zeros([40, 128], device=device, dtype=torch.float32)

    matmul_wide_alias(x, y, out)
    torch.npu.synchronize()

    ref = x[:40, :].float()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"wide alias matmul passed, max diff = {(out - ref).abs().max().item()}")
```
