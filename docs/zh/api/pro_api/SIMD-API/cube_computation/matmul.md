# pypto_pro.language.matmul

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

对lhs_tile和rhs_tile执行矩阵乘法，并将结果写入dst_tile：

```text
dst_tile = lhs_tile × rhs_tile
```

## 函数原型

```python
pypto_pro.language.matmul(
    dst_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    bias_tile: Optional[Tile] = None,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L0C Buffer，形状为[M, N]，layout必须为NZ。数据类型为DT_FP32或DT_INT32时，fractal未指定则自动设为1024。支持通过valid_shape或pypto_pro.language.set_validshape设置尾块的有效形状，有效M、N必须与实际矩阵乘结果范围一致。支持的数据类型组合详见[约束说明](#约束说明)。 |
| lhs_tile | 输入 | 源操作数（左矩阵），Tile类型，存储空间为L0A Buffer，形状为[M, K]，layout必须为NZ，K必须与rhs_tile的K维一致。支持的数据类型组合详见[约束说明](#约束说明)。 |
| rhs_tile | 输入 | 源操作数（右矩阵），Tile类型，存储空间为L0B Buffer，形状为[K, N]，layout必须为ZN，K必须与lhs_tile的K维一致。支持的数据类型组合详见[约束说明](#约束说明)。 |
| bias_tile | 输入 | 源操作数（可选偏置），Tile类型，存储空间为BiasTable Buffer，形状为[1, N]，layout默认且仅支持ND。偏置沿M维广播，数据类型必须与dst_tile一致。传入本参数时，在Fixpipe阶段融合偏置加法，无需额外调用pypto_pro.language.add。只能作为第四个位置参数传入。 |
| phase | 输入 | 可选，开启硬件unitFlag机制，[pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md)类型。与[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)的配合方式见[Cube计算进阶](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md)。 |

## 约束说明

- 数据类型约束：

  | lhs_tile（L0A Buffer） | rhs_tile（L0B Buffer） | dst_tile（L0C Buffer） |
  |---|---|---|
  | DT_INT8 | DT_INT8 | DT_INT32 |
  | DT_FP16 | DT_FP16 | DT_FP32 |
  | DT_BF16 | DT_BF16 | DT_FP32 |
  | DT_FP32 | DT_FP32 | DT_FP32 |
  | DT_HF8 | DT_HF8 | DT_FP32 |
  | DT_FP8E4M3FN或DT_FP8E5M2 | DT_FP8E4M3FN或DT_FP8E5M2 | DT_FP32 |

- 使用bias_tile的matmul会初始化L0C Buffer，只能用于K维分块的首块；后续分块必须使用matmul_acc累加。
- K维分块时，首块使用matmul初始化L0C Buffer，后续块使用matmul_acc累加。启用phase后，非末块使用pypto_pro.language.AccPhase.Partial，末块使用pypto_pro.language.AccPhase.Final，并与pypto_pro.language.store、pypto_pro.language.store_tile及pypto_pro.language.move的[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)配合使用。

## 返回值说明

无。

## 调用示例

### 单次matmul（无bias、无K维分块）

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128      # K 恰好一个 Tile，无需分块累加
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A Buffer、L0B Buffer和L0C Buffer均使用单Tile组（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):          # M 维分块
            for j in pl.range(0, N_SIZE, TILE):      # N 维分块
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### 带偏置的matmul（无K维分块）

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_bias_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    bias: pl.Tensor[[1, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP16],
):
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[6])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[7])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[8])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[9])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):
            for j in pl.range(0, N_SIZE, TILE):
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                cur_bias_l1 = bias_l1.next()
                al = a_left.current()
                br = b_right.current()
                bl = bias_l0b.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.load(cur_bias_l1, bias, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.move(bl, cur_bias_l1)             # L1 Buffer的FP16数据搬入BiasTable Buffer并转换为FP32
                pl.matmul(ac, al, br, bl)             # out = A @ B + bias
                pl.store(c, ac, [i, j])
```

### 带偏置的K维分块累加

```python
import pypto_pro.language as pl

TILE = 128
K_SPLIT = 384     # 分 3 个 TILE 块


@pl.jit(auto_mutex=True)
def matmul_k_split_bias_kernel(
    a: pl.Tensor[[TILE, K_SPLIT], pl.DT_FP16],
    b: pl.Tensor[[K_SPLIT, TILE], pl.DT_FP16],
    bias: pl.Tensor[[1, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP16],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[6, 7])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[8, 9])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[10, 11])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[12])

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, K_SPLIT, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                # 首块：加载 bias，matmul 覆盖写入 acc = A0@B0 + bias
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
```
