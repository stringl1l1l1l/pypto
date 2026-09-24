# pypto_pro.language.insert

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

把一块较小的源Tile，按offset=[row, col]指定的行列位置嵌入到一块较大的目标Tile中。建议使用[pypto_pro.language.move](move.md)接口代替insert接口。

## 函数原型

```python
pypto_pro.language.insert(dst_tile: Tile, src_tile: Tile, offset: List[int]) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tile | 输入 | 源操作数，Tile类型，支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| offset | 输入 | 位置偏移，List[int]类型，长度必须为2，格式为[row, col]。源Tile的左上角对齐到目标Tile的offset位置；offset[0]为目标Tile的行偏移row，offset[1]为列偏移col。row和col必须是非负整数或运行时整数表达式，且源Tile的有效区域必须完整落入目标Tile。 |

## 约束说明

- 数据类型及分形约束：

  | 源 → 目的 | 分形要求 | 数据类型要求 |
  |---|---|---|
  | UB → UB | ND → ND、NZ → NZ。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | UB → L1 Buffer | 源支持ND、NZ，目的不校验分形。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | L0C Buffer → UB | NZ → ND，NZ → DN，NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → L1 Buffer | NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |

  当源操作数在UB且为ND格式，目的操作数在L1 Buffer且为NZ或ZN格式时，避免直接进行数据搬运，必须先通过pypto_pro.language.move将源操作数搬运至另一块NZ格式的UB，再将其搬运至目的操作数，正确用法参见[将UB中的计算结果拼接到L1 Buffer](#将ub中的计算结果拼接到l1-buffer)。

## 返回值说明

无。

## 调用示例

### 将UB中的计算结果拼接到L1 Buffer

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def insert_matmul_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x10000, mutex_ids=[0])

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000, mutex_ids=[1])
        tile_y_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x2000, mutex_ids=[2])
        tile_sum_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000, mutex_ids=[3])
        tile_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                             layout=pl.NZ),
            addrs=0x6000, mutex_ids=[4])
        v1_mat = v1_mat_group.current()
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        tile_sum = tile_sum_group.current()
        tile_nz = tile_nz_group.current()

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])

        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)   # ND -> NZ

        pl.insert(v1_mat, tile_nz, [off, 0])   # UB -> L1 NZ2NZ
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[5])
        v1_left_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[6])
        rhs_right_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN),
            addrs=0x0000, mutex_ids=[7])
        c_l0c_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024),
            addrs=0x0000, mutex_ids=[8])
        v1_mat = v1_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        v1_left = v1_left_group.current()
        rhs_right = rhs_right_group.current()
        c_l0c = c_l0c_group.current()

        pl.load(rhs_mat, rhs, [0, 0])
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.store(out, c_l0c, [0, 0])
```

### 其他二维偏移场景

```python
# 两个维度均有偏移
pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, TS_HALF * sub_id])

# 仅沿第 0 维偏移
pl.insert(v1_mat, tile_nz, [off, 0])
```
