# pypto_pro.language.load

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

将GM Tensor中的数据搬入L1 Buffer或UB中的Tile。offsets按Tensor各维的绝对元素坐标指定搬运起始位置；Tensor维数大于Tile维数时，可通过order指定Tile的两个维度与Tensor维度的对应关系，并控制是否转置。

如果需要按Tile块编号指定搬运位置，使用[pypto_pro.language.load_tile](load_tile.md)。

## 函数原型

```python
pypto_pro.language.load(
    dst_tile: Tile,
    src_tensor: Tensor,
    offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L1 Buffer或UB，首地址必须按32字节对齐。接口按照该Tile的valid_shape搬运数据。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tensor | 输入 | 源操作数，Tensor类型，存储空间为GM。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| offsets | 输入 | 表示源Tensor各维度的绝对元素坐标，List[int或Scalar]类型，长度须与源Tensor的维数相同。<br>- 不支持负数。<br>- 未被order选中的维度使用对应值固定下标。<br>- 搬运起始位置必须位于源Tensor的shape范围内；边界Tile可通过pypto_pro.language.set_validshape设置有效形状，保证有效搬运范围不越过源Tensor边界。<br>- 例如，offsets=[64, 0]表示从二维Tensor的第64行、第0列开始搬运。 |
| order | 输入 | 可选，维度映射，长度为2的编译期整数列表。列表中的第i项表示dst_tile第i维对应src_tensor的维度，两个维度索引必须互不重复且位于src_tensor的维度范围内。<br>- 升序表示不转置，例如order=[0, 1]。<br>- 降序表示转置，例如order=[1, 0]。<br>- 普通Tensor不设置order时，dst_tile默认对应src_tensor的最后两个维度，即[ndim - 2, ndim - 1]，且不转置。<br>- 搬运MX矩阵乘量化系数时，src_tensor的最后一维是物理phase轴，不能在order中选择；不设置order时，默认对应phase轴之前的两个维度，即[ndim - 3, ndim - 2]。 |

## 约束说明

- 数据类型及分形约束：

  src_tensor显式声明为NZ时，按NZ搬运；其他场景下，order升序时按ND搬运，order降序时按DN搬运。支持的组合如下。

  | 源 → 目的 | 分形要求 | 数据类型要求 |
  |---|---|---|
  | GM → UB | 源与目的分形必须相同，支持ND、DN、NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | ND → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | DN → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8。 |
  | GM → L1 Buffer | DN → ZN、NZ → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | ND → ND。 | 源与目的数据类型必须相同，仅支持DT_INT64、DT_UINT64。 |
  | GM → L1 Buffer | ND/DN → ZZ/NN，仅用于pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc的量化系数搬运。 | 源与目的数据类型必须为DT_FP8E8M0。 |

  GM → UB的DN → DN搬运仅支持目标Tile为[N, 1]的单列形态。

- NZ搬运约束：

  - src_tensor声明为NZ时，dst_tile必须为NZ，且只支持从src_tensor的最后两个维度正序搬运，不支持通过降序order转置。NZ的物理排布和Tensor shape约束详见[TensorLayout](../basic_data_structures/TensorLayout.md)。
  - dst_tile的shape和valid_shape中，M必须按16对齐，N必须按src_tensor数据类型对应的C0对齐；N方向的offset也必须按C0对齐。

- MX矩阵乘量化系数搬运约束：

  - 仅支持将DT_FP8E8M0的src_tensor搬入fractal为32、布局为ZZ或NN的L1 Buffer Tile，并作为pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc的量化系数使用。
  - src_tensor的维数必须大于等于3，最后一维为长度等于2的物理phase轴，offsets中phase轴对应的偏移必须为0。显式设置order时，不能选择phase轴。

- Tile地址复用约束：

  开启auto_mutex时，如果连续两次pypto_pro.language.load写入同一个UB或L1 Buffer地址，且两次搬运之间没有操作读取前一次搬入的数据，需要在两次搬运之间调用[pypto_pro.language.system.bar_mte2](../synchronization/bar_mte2.md)。pypto_pro.language.system.bar_mte2仅保证两次写操作的先后顺序；如果后续仍需使用前一次搬入的数据，应在复用地址前先读取或复制该数据。

## 返回值说明

无。

## 调用示例

### 按绝对元素坐标搬运

将输入Tensor中行索引为64～127的数据搬入UB Tile，计算后写入输出Tensor中行索引为0～63的位置。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
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
        pl.load(current_x, x, [64, 0])
        pl.add(current_out, current_x, current_x)
        pl.store(out, current_out, [0, 0])
```

### 从高维Tensor中选择维度搬运

高维Tensor的维度映射方式如下。

| Tensor shape | Tile对应的Tensor维度 | order | offsets示例 |
|---|---|---|---|
| [B, N, S, D] | S、D | 不设置，默认值为[2, 3] | [b, n, s_offset, 0] |
| [B, S, N, D] | S、D | [1, 3] | [b, s_offset, n, 0] |

以下示例显式设置order=[1, 3]，offsets中的第0项和第2项分别固定B、N的下标。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_high_dim_kernel(
    x: pl.Tensor[[2, 64, 8, 128], pl.DT_FP16],
    out: pl.Tensor[[2, 64, 8, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        current_x = x_tile.current()
        current_out = out_tile.current()
        # 固定B=1、N=3，从S=0、D=0开始搬运。
        pl.load(current_x, x, [1, 0, 3, 0], order=[1, 3])
        pl.add(current_out, current_x, current_x)
        pl.store(out, current_out, [1, 0, 3, 0], order=[1, 3])
```

### Cube侧转置搬运

Cube计算中，左矩阵搬入L0A Buffer时使用shape=[M, K]、NZ，右矩阵搬入L0B Buffer时使用shape=[K, N]、ZN。L1 Buffer Tile的shape需要与对应的L0A Buffer或L0B Buffer Tile一致；从普通ND GM Tensor搬入L1 Buffer时，是否转置由order和L1 Buffer Tile的分形共同决定。

| 矩阵 | 是否转置 | GM Tensor shape | L1 Buffer Tile | order |
|---|---|---|---|---|
| 左矩阵A[M, K] | 否 | [M, K] | shape=[M, K]，NZ | 不设置，默认值为[0, 1] |
| 左矩阵A[M, K] | 是 | [K, M] | shape=[M, K]，ZN | [1, 0] |
| 右矩阵B[K, N] | 否 | [K, N] | shape=[K, N]，NZ | 不设置，默认值为[0, 1] |
| 右矩阵B[K, N] | 是 | [N, K] | shape=[K, N]，ZN | [1, 0] |

order=[1, 0]表示Tile的第0维对应Tensor的第1维、Tile的第1维对应Tensor的第0维，搬运时执行转置。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_left_transpose_kernel(a_transposed: pl.Tensor[[128, 64], pl.DT_FP16]):
    tile_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
    )
    a_l1 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_cube():
        pl.load(a_l1.current(), a_transposed, [0, 0], order=[1, 0])
```

### 搬运尾块

输入Tensor的剩余数据小于Tile的shape时，需要根据后续操作是否读取无效区域分别处理：

- pypto_pro.language.set_validshape用于设置当前Tile的有效形状，使pypto_pro.language.load仅搬运valid_shape范围内的数据，避免越界读取GM。
- 如果后续操作会读取valid_shape以外的区域，例如归约或矩阵乘，需要在TileType中设置pad，并调用pypto_pro.language.fillpad将无效区域填充为安全值。求和时填充zero，取最大值或softmax时填充min，取最小值时填充max。

#### 后续操作不读取无效区域

以下示例只将尾块写回GM，后续操作不会读取无效区域，因此不需要调用pypto_pro.language.fillpad。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tail_kernel(
    x: pl.Tensor[[70, 128], pl.DT_FP16],
    out: pl.Tensor[[70, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 128],
        valid_shape=[-1, -1],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        current_x = x_tile.current()
        pl.set_validshape(current_x, [6, 128])
        pl.load(current_x, x, [64, 0])
        pl.store(out, current_x, [64, 0])
```

#### 后续操作读取无效区域

后续操作会读取整个Tile时，需要把有效数据搬入带动态valid_shape的源Tile，再通过pypto_pro.language.fillpad写入配置了pad的目的Tile。以下示例将输入Tensor末尾6行搬入Tile，并把其余区域填充为0；补零后的Tile可用于求和等需要读取整个Tile的操作。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tail_with_padding_kernel(
    x: pl.Tensor[[70, 128], pl.DT_FP16],
    padded_out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    src_type = pl.TileType(
        shape=[64, 128],
        valid_shape=[-1, -1],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    padded_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )
    src_tile = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    padded_tile = pl.make_tile_group(type=padded_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        current_src = src_tile.current()
        current_padded = padded_tile.current()
        pl.set_validshape(current_src, [6, 128])
        pl.load(current_src, x, [64, 0])
        pl.fillpad(current_padded, current_src)
        pl.store(padded_out, current_padded, [0, 0])
```

多核任务中尾块数量、有效形状和任务索引的计算方法，详见[尾块处理](../../../../guide/programming_guide/pro/development/tiling/multi_core_tiling.md#尾块处理)。

### 复用Tile地址

开启auto_mutex并连续复用同一个UB或L1 Buffer地址时，根据前一次搬入的数据是否已经被读取决定是否需要手动同步。

```python
import pypto_pro.language as pl


# 前一次搬入的数据未被读取：复用地址前需要等待MTE2搬运完成。
pl.load(input_tile, x, [0, 0])
pl.system.bar_mte2()
pl.load(input_tile, x, [64, 0])

# 前一次搬入的数据已被Vector计算读取：数据依赖已经建立同步，
# 下一次搬运可以直接复用input_tile的地址。
pl.load(input_tile, x, [0, 0])
pl.add(output_tile, input_tile, input_tile)
pl.load(input_tile, x, [64, 0])
```
