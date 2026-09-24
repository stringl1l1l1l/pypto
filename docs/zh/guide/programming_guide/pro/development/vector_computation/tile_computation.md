# Tile计算

Tile计算是PyPTO Pro在Vector侧提供的矢量计算方式。开发者以UB中的Tile为输入和输出，通过Tile计算接口对有效区域中的多个数据元素执行批量计算，适合逐元素运算、归约、类型转换和数据重排等规则计算。

一个完整算子的典型数据链路如下：

```text
Tensor → 数据搬入 → 输入Tile → Tile计算 → 输出Tile → 数据搬出 → Tensor
```

Tile及TileGroup的创建、片上地址和数据搬运等内容由[Tile创建和操作](../tile_creation_and_operations.md)介绍。

## 计算模型

Tile计算在Vector执行域中执行。调用计算接口前需要准备输入Tile和输出Tile，接口将结果写入输出Tile，不会创建或返回新的Tile：

```python
import pypto_pro.language as pl

with pl.section_vector():
    pl.add(out_tile, lhs_tile, rhs_tile)
```

上例对lhs_tile和rhs_tile的有效区域逐元素相加，并将结果写入out_tile。多数Tile计算接口将输出Tile作为第一个参数，其后依次传入输入、临时Tile和可选参数。

- out是保存计算结果的目的Tile，需要在调用接口前准备完成。
- 输入可以是Tile；部分接口也支持Scalar或Python标量常量。
- 计算作用于Tile的有效区域，输入输出的shape、valid_shape、数据类型和layout需要满足对应接口的约束。
- 部分接口支持原地计算，即out可以与某个输入使用同一Tile；未明确说明时不能假定支持原地计算。

## 计算接口与常用模式

PyPTO Pro当前公开的Tile计算操作如下：

| 计算类型 | 主要用途 |
|---|---|
| [逐元素计算](../../../../../api/pro_api/SIMD-API/tile_computation/elementwise/index.md) | 对Tile中的对应元素执行算术、逻辑或激活计算。 |
| [比较](../../../../../api/pro_api/SIMD-API/tile_computation/comparison/index.md) | 逐元素比较并生成按位压缩的掩码Tile。 |
| [选择](../../../../../api/pro_api/SIMD-API/tile_computation/selection/index.md) | 根据掩码从两个输入中逐元素选择结果。 |
| [类型转换](../../../../../api/pro_api/SIMD-API/tile_computation/type_conversion/index.md) | 将源Tile有效区域中的元素转换为目的Tile的数据类型。 |
| [数学函数](../../../../../api/pro_api/SIMD-API/tile_computation/math_functions/index.md) | 使用标量填充Tile，或沿指定维度执行求和归约。 |
| [复合计算](../../../../../api/pro_api/SIMD-API/tile_computation/composite_computation/index.md) | 完成Tile与标量的乘加计算。 |
| [融合矢量计算](../../../../../api/pro_api/SIMD-API/tile_computation/fused_vector_computation/index.md) | 在一次接口调用中完成加法和ReLU激活。 |
| [转置](../../../../../api/pro_api/SIMD-API/tile_computation/transpose_and_element_access/index.md) | 交换二维Tile的两个轴。 |

各接口支持的数据类型、存储空间、layout和原地计算方式可能不同，应以[Tile计算API](../../../../../api/pro_api/SIMD-API/tile_computation/index.md)中的说明为准。

### Tile与Tile逐元素计算

两个输入Tile的对应元素参与计算，结果写入目的Tile：

```python
import pypto_pro.language as pl

pl.add(out_tile, lhs_tile, rhs_tile)
pl.mul(out_tile, lhs_tile, rhs_tile)
pl.maximum(out_tile, lhs_tile, rhs_tile)
```

使用这类接口前，需要根据对应API确认输入输出Tile的shape、valid_shape、数据类型和layout是否匹配。

### Tile与Scalar逐元素计算

部分接口支持Tile与Scalar计算，同一个Scalar会参与Tile有效区域中每个元素的运算。例如：

```python
import pypto_pro.language as pl

pl.add(out_tile, src_tile, 1.0)
```

Tile-Scalar并不是所有逐元素接口的通用能力，Scalar类型也需要与Tile元素类型兼容。

### 原地计算

接口明确支持原地计算时，可以复用输入Tile保存结果。例如，以下代码依次完成加法和ReLU：

```python
import pypto_pro.language as pl

pl.add(out_tile, lhs_tile, rhs_tile)
pl.relu(out_tile, out_tile)
```

原地计算可以减少中间Tile，但会覆盖原数据。如果后续计算仍需使用输入值，应使用独立的输出Tile。

### 比较与选择

比较接口生成按位压缩的掩码，选择接口根据掩码选择结果：

```python
import pypto_pro.language as pl

pl.eq(mask_tile, lhs_tile, rhs_tile)
pl.select(out_tile, mask_tile, lhs_tile, fallback_tile, tmp_tile)
```

掩码Tile、临时Tile以及输入输出Tile的类型和layout需要满足对应接口的要求。

### 归约计算

归约接口可以沿指定维度对二维Tile求和：

```python
import pypto_pro.language as pl

pl.sum(out_tile, src_tile, tmp_tile, dim=0)
pl.sum(out_tile, src_tile, tmp_tile, dim=1)
```

这里的dim表示归约方向，而不是直接删除同编号的轴。源Tile形状为[M, N]时，dim=0对每一行沿N方向求和，输出形状为[M, 1]；dim=1对每一列沿M方向求和，输出形状为[1, N]。不同归约方向支持的数据类型存在差异。

### 类型与布局转换

类型转换接口将Tile转换为目的Tile的数据类型，目标类型由out_tile.dtype决定；转置接口交换二维Tile的两个轴：

```python
import pypto_pro.language as pl

pl.cast(out_tile, src_tile, mode=pl.RoundMode.CAST_ROUND)
pl.transpose(transposed_tile, src_tile)
```

这两个接口都将结果写入预先准备的目的Tile。舍入模式、数据类型组合以及shape与对齐要求参见对应API说明。

## 计算编排与同步

一次Tile计算可以按照以下思路组织：

1. **确定计算语义**：明确计算属于逐元素、比较选择、归约、类型转换还是数据重排，并选择对应的Tile计算接口。
2. **确定输入输出关系**：根据接口原型准备输入、输出和临时Tile，确认shape、valid_shape、数据类型及layout满足接口约束。
3. **编排计算操作**：在Vector执行域中调用一个或多个Tile计算接口，将前一个操作的输出Tile作为后一个操作的输入Tile。
4. **处理数据依赖**：当Tile在数据搬运流水与Vector计算流水之间传递时，使用基于TileGroup mutex元数据的自动同步或显式同步，保证生产者先于消费者完成。
5. **处理尾块**：当实际数据范围小于Tile物理shape时，设置正确的valid_shape，使计算只作用于有效区域。

多个Tile计算接口可以在Vector执行域中组成计算链。前一个接口的输出可作为后一个接口的输入；同一Vector计算流水上的相邻操作按依赖顺序执行，不需要在操作之间插入跨Pipe同步。Tile在MTE2、Vector和MTE3等不同流水之间传递时，仍需通过基于TileGroup mutex元数据的自动同步或[显式同步](../../../../../api/pro_api/SIMD-API/synchronization/index.md)保证读写顺序。

以下片段先对两个输入Tile逐元素相加，再原地执行ReLU。代码假设Tile已经创建、输入数据已经搬入，计算前后的跨Pipe依赖由外层Kernel处理：

```python
import pypto_pro.language as pl

with pl.section_vector():
    pl.add(out_tile, lhs_tile, rhs_tile)
    pl.relu(out_tile, out_tile)
```

如果改用[pypto_pro.language.add_relu](../../../../../api/pro_api/SIMD-API/tile_computation/fused_vector_computation/add_relu.md)，需要注意该接口会在计算过程中修改lhs_tile。只有后续不再使用其原始数据，且输入输出满足接口约束时，才能用融合接口替代前述计算链。

## 尾块处理

当当前数据块的实际有效范围小于Tile的物理shape时，该Tile称为**尾块**。最常见的情况是GM Tensor的行数或列数不能被Tile尺寸整除，也可能来自对Tensor子区域或运行时有效窗口的处理。本节介绍如何使用valid_shape限定尾块的有效区域，以及何时需要填充无效区域。

### 识别尾块

以二维Tensor为例，设Tensor shape为[M, N]，Tile shape为[TILE_M, TILE_N]，两个方向的Tile数量为：

```python
m_tiles = (M + TILE_M - 1) // TILE_M
n_tiles = (N + TILE_N - 1) // TILE_N
```

当M % TILE_M != 0时会产生尾行，当N % TILE_N != 0时会产生尾列；两者同时出现时，右下角为尾角。

**图1 二维Tensor中的满块、尾行、尾列和尾角**

![二维Tensor中的满块、尾行、尾列和尾角](../../../../figures/pro/pro_tail_tile_grid.png)

对于第i行、第j列Tile，当前有效行列数可按下式计算：

```python
import pypto_pro.language as pl

valid_rows = pl.min(M - i * TILE_M, TILE_M)
valid_cols = pl.min(N - j * TILE_N, TILE_N)
```

### shape与valid_shape

TileType.shape和TileType.valid_shape描述的对象不同：

| 参数 | 作用 |
| --- | --- |
| shape | Tile的物理规格，决定片上缓冲区大小和寻址边界 |
| valid_shape | 当前Tile中真正有效的行列范围 |

**图2 Tile物理shape与逻辑valid_shape的关系**

![Tile物理shape与逻辑valid_shape的关系](../../../../figures/pro/pro_tail_shape_validshape.png)

对于每次运行时有效尺寸可能不同的尾块，[TileType](../../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)可省略valid_shape；默认值None等同于动态模式[-1, -1]：

```python
import pypto_pro.language as pl

tile_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)
```

省略valid_shape时，两个维度的有效大小都可在运行时确定，无需显式写出[-1, -1]。如果只有一个维度动态，也可以声明为valid_shape=[64, -1]。运行时传给pypto_pro.language.set_validshape的有效大小必须为正整数，且不能超过shape的对应维度。

### 标准处理流程

尾块处理的关键顺序是：**先设置有效形状，再搬入和计算**。

**图3 尾块处理流程**

![尾块的计算、有效形状设置、搬入、计算和搬出流程](../../../../figures/pro/pro_tail_processing_flow.png)

```python
import pypto_pro.language as pl

tile_a = a_group.next()
tile_b = b_group.next()
tile_c = c_group.next()

valid_rows = pl.min(M - i * TILE_M, TILE_M)
valid_cols = pl.min(N - j * TILE_N, TILE_N)

# 必须先设置，使随后的 load、计算和 store 使用同一有效区。
pl.set_validshape(tile_a, [valid_rows, valid_cols])
pl.set_validshape(tile_b, [valid_rows, valid_cols])
pl.set_validshape(tile_c, [valid_rows, valid_cols])

pl.load_tile(tile_a, a, [i, j])
pl.load_tile(tile_b, b, [i, j])
pl.add(tile_c, tile_a, tile_b)
pl.store_tile(c, tile_c, [i, j])
```

pypto_pro.language.set_validshape传入Tile时，只更新该Tile的有效区域；传入TileGroup时，会更新组内所有Tile的有效区域。在上述顺序中：

- 数据搬入操作只从GM搬入有效区域，避免尾块越界读。
- 矢量计算使用当前有效区域。
- 数据写回操作只写回有效区域，避免越界写。

> 应在数据搬入前设置valid_shape，用于约束GM搬入；在数据搬入后设置仅影响后续操作。

#### Tile与TileGroup

每个缓冲区的有效形状不同时，应分别为从TileGroup中获取的Tile设置valid_shape：

```python
import pypto_pro.language as pl

tile = tile_group.next()
pl.set_validshape(tile, [valid_rows, valid_cols])
```

如果TileGroup中的所有缓冲区在整个Kernel期间都使用同一有效形状，可以对TileGroup统一设置：

```python
import pypto_pro.language as pl

pl.set_validshape(tile_group, [valid_rows, valid_cols])
```

对逐块变化的尾块，应在每次获取Tile后设置当前块的有效形状。

### compact与Tile计算

valid_shape描述Tile中参与当前操作的有效区域；compact描述特定搬运、重排或矩阵计算路径如何解释片上物理排布，两者不能互相替代。

本节的UB ND逐元素尾块使用valid_shape控制实际搬运和计算范围，片上跨度仍由物理shape确定，通常不需要配置compact。涉及分形搬运、Cube计算或RowPlusOne排布时，应按照具体接口要求配置，详见[TileType的compact参数](../../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md#参数说明)和[Cube计算](../cube_computation.md)。

### 何时需要填充无效区域

valid_shape只标记哪些元素有效，不会自动给无效区域写入数值。如果后续操作会读取整个物理Tile，需要使用pad指定安全填充值，并显式执行填充操作。

| 计算语义 | 建议填充值 |
| --- | --- |
| 逐元素加、减、乘，且计算与写回都遵循valid_shape | 通常不需要填充 |
| 求和 | 0 |
| 求最大值或Softmax前的最大值归约 | 数据类型最小值 |
| 求最小值 | 数据类型最大值 |

下面的关键片段将src的无效区域填为0，TileGroup的创建过程已省略：

```python
import pypto_pro.language as pl

src_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)
dst_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    pad=pl.TilePad.zero,
)

src = src_group.next()
dst = dst_group.next()
pl.set_validshape(src, [valid_rows, valid_cols])
pl.load(src, x, [row_offset, col_offset])
pl.fillpad(dst, src)
```

pad只指定填充语义，[pypto_pro.language.fillpad](../../../../../api/pro_api/SIMD-API/memory_data_movement/load.md#搬运尾块)才会执行填充。是否需要填充取决于后续操作会不会读取无效区域，以及无效值是否影响计算结果。矩阵计算尾块的处理方式参见[Cube计算](../cube_computation.md)。

### 完整示例：二维加法的四类尾块

下面的Kernel支持动态二维形状。x、y和z的shape必须相同；Kernel使用x.shape计算循环范围，并以相同的块索引访问另外两个Tensor。示例使用64 × 128的物理Tile，通过逐块计算valid_rows和valid_cols，同时处理满块、尾行、尾列和尾角。

```python
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 64
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_tail_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    a_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
    b_group = pl.make_tile_group(
        type=tile_type, addrs=[0x8000, 0xC000], mutex_ids=[2, 3])
    c_group = pl.make_tile_group(
        type=tile_type, addrs=[0x10000, 0x14000], mutex_ids=[30, 31])

    with pl.section_vector():
        m = x.shape[0]
        n = x.shape[1]
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N

        for i in pl.range(0, m_tiles, 1):
            for j in pl.range(0, n_tiles, 1):
                tile_a = a_group.next()
                tile_b = b_group.next()
                tile_c = c_group.next()

                valid_rows = pl.min(m - i * TILE_M, TILE_M)
                valid_cols = pl.min(n - j * TILE_N, TILE_N)
                pl.set_validshape(tile_a, [valid_rows, valid_cols])
                pl.set_validshape(tile_b, [valid_rows, valid_cols])
                pl.set_validshape(tile_c, [valid_rows, valid_cols])

                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])

device = "npu:0"
torch.npu.set_device(device)
x = torch.randn(129, 257, dtype=torch.float16, device=device)
y = torch.randn_like(x)
z = torch.empty_like(x)
add_tail_kernel[None, 1](x, y, z)
torch.npu.synchronize()
torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-3)
```

129 × 257在两个维度上均包含尾块，因此示例覆盖满块、行尾块、列尾块和角尾块四类Tile，并分别设置对应的valid_shape。

### 常见问题

#### 在数据搬入后设置valid_shape

```python
import pypto_pro.language as pl

# 错误：本次 load 已经发生，无法再用 valid_shape 限制它。
pl.load(tile, x, offsets)
pl.set_validshape(tile, [valid_rows, valid_cols])
```

应调整为：

```python
import pypto_pro.language as pl

pl.set_validshape(tile, [valid_rows, valid_cols])
pl.load(tile, x, offsets)
```

#### 测试只覆盖满块

例如物理Tile为[64, 128]，测试仍使用[64, 128]的Tensor，只能证明满块路径可用，不能证明尾块不越界。尾块测试应至少包含一组两个维度均小于物理Tile的形状，或一组两个维度均不能整除Tile尺寸的较大形状。

#### 只给输入Tile设置valid_shape

输入Tile、计算结果Tile和写回Tile应对同一逻辑区域使用一致的有效形状。遗漏输出Tile可能导致越界写或写回无效数据。

#### 把pad当成自动填充

pad只声明填充值，不会单独产生填充操作。需要对无效区域进行实际填充时，应显式调用填充接口。

#### 对所有尾块执行填充

逐元素计算通常只需要正确设置有效形状。只有后续操作会读取无效区域，且无效值会影响结果时，才需要选择与计算语义匹配的填充值。

### 参数选择速查

| 场景 | valid_shape | compact | 填充方式 |
| --- | --- | --- | --- |
| 固定形状且全部为满块 | 与shape一致 | 不需要 | 不需要 |
| 矢量逐元素ND动态尾块 | [-1, -1]，逐块设置 | 不需要 | 通常不需要 |
| 尾块后执行求和 | [-1, -1] | 不需要 | 填充0 |
| 尾块后执行最大值归约 | [-1, -1] | 不需要 | 填充数据类型最小值 |
| Cube动态尾块 | 为参与矩阵乘的L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer中的Tile设置匹配的有效尺寸 | L0A Buffer、L0B Buffer和L0C Buffer通常设为1；特殊路径按接口要求配置 | 由具体数据路径和算子语义决定 |

## 计算约束与建议

- Tile计算接口通常以输出Tile作为第一个参数，计算结果写入该Tile，不能按Tensor表达式的方式接收返回值。
- 不同接口对shape、valid_shape、数据类型、layout和内存空间的要求不同，组合接口时需要同时满足前后两个操作的约束。
- 只在API明确支持时使用Tile-Scalar、原地计算或输入输出复用。
- 部分选择和归约接口需要额外的掩码或临时Tile，应在设计片上空间时一并考虑。
- 计算前后的跨Pipe依赖必须正确同步，具体接口参见[同步控制](../../../../../api/pro_api/SIMD-API/synchronization/index.md)。

Tile计算的完整接口列表参见[Tile计算API](../../../../../api/pro_api/SIMD-API/tile_computation/index.md)。
