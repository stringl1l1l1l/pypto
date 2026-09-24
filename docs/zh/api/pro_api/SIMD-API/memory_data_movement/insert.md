# pypto_pro.language.insert

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

将较小的源Tile搬运到较大的目标Tile中由offset=[row, col]指定的位置，并可在搬运过程中实现随路格式转换、量化和激活等操作。建议使用[pypto_pro.language.move](move.md)接口代替insert接口。

## 函数原型

```python
pypto_pro.language.insert(
    dst_tile: Tile,
    src_tile: Tile,
    offset: List[int],
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    phase: Optional[STPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型；L0C Buffer → L1 Buffer路径必须使用NZ布局，首地址必须32字节对齐。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tile | 输入 | 源操作数，Tile类型，支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| offset | 输入 | 位置偏移，List[int]类型，长度必须为2，格式为[row, col]，单位为元素个数。源Tile的左上角对齐到目标Tile的offset位置；row和col必须是非负整数或运行时整数表达式，并满足row + src行数 ≤ dst行数、col + src列数 ≤ dst列数。 |
| relu_pre_mode | 输入 | 可选；L0C Buffer → L1 Buffer搬运时是否开启随路ReLU操作，[pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md)类型。其他路径不支持。 |
| scale | 输入 | 可选，仅用于L0C Buffer → L1 Buffer搬运，设置随路量化参数。数据在搬出L0C Buffer时由Fixpipe乘以该比例并转换到目的数据类型。不同的传入形式会影响量化粒度，支持如下类型：<br>- **float类型**：直接传入固定值（如scale = 2.0），适用于整块Tile使用同一比例。<br>- **Scalar类型**：量化比例在运行时确定，需按数据类型传值。<br>&nbsp;&nbsp;- DT_FP32：直接传原始比例值（如0.5）。<br>&nbsp;&nbsp;- DT_INT32、DT_INT64：传预编码的float32位模式转成的整数（如struct.pack("!f", 0.5)）。<br>- **Tile类型**：每列使用独立比例，需满足以下要求：<br>&nbsp;&nbsp;- 目标存储区域必须为Fixpipe Buffer。<br>&nbsp;&nbsp;- shape为[1, N]（列量化），N必须是16的倍数且N ≤ 512。<br>&nbsp;&nbsp;- dtype为DT_INT64。<br>&nbsp;&nbsp;- 目的操作数的Tile数据类型为DT_INT8时，Fixpipe Buffer中的Tile每个DT_INT64元素的bit46需置1，用于选择有符号量化；未置位时L0C Buffer中的负值会被按无符号解读。<br>&nbsp;&nbsp;- 用户需要先把比例数据从GM搬到L1 Buffer，再搬到Fixpipe Buffer，并完成MTE1→FIX同步。 |
| phase | 输入 | 可选，L0C Buffer → L1 Buffer搬运时是否启用unit_flag同步，详见[Cube计算进阶](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md)。 |

## 约束说明

- 数据类型及分形约束：

  | 源 → 目的 | 分形要求 | 数据类型要求 |
  |---|---|---|
  | UB → UB | ND → ND、NZ → NZ。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | UB → L1 Buffer | 源支持ND、NZ，目的不校验分形。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | L0C Buffer → UB | NZ → ND、NZ → DN、NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → L1 Buffer（不配置scale） | NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → L1 Buffer（配置scale） | NZ → NZ。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16/DT_HF8/DT_FP8E4M3FN，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |

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

# L0C FP32 NZ子块随路量化后写入L1 INT8 NZ目标窗口
pl.insert(mat_int8, acc_fp32, [16, 32], scale=2.0, relu_pre_mode=pl.ReluPreMode.NormalRelu)
```
