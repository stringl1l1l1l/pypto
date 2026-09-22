# pypto_pro.language.store

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

把UB或L0C Buffer中的Tile按绝对元素坐标写回GM，与[pypto_pro.language.load](load.md)对应。数据搬运过程中支持随路ReLU、量化或原子累加等操作。

## 函数原型

```python
pypto_pro.language.store(
    dst_tensor: Tensor,
    src_tile: Tile,
    offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tensor | 输出 | 目的操作数，Tensor类型，存储空间为GM。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tile | 输入 | 源操作数，Tile类型，存储空间为UB或L0C Buffer。<br>- UB Tile的首地址须按32字节对齐；<br>- L0C Buffer Tile的首地址须按64字节对齐。 |
| offsets | 输入 | 可选，表示目的Tensor各维度的绝对元素坐标，List[int或Scalar]类型，长度须与目的Tensor的维数相同。<br>- 不支持负数。<br>- 对于高维NZ Tensor，最后两项对应M、N方向。 |
| relu_pre_mode | 输入 | 可选，L0C Buffer→GM搬运时是否开启随路ReLU操作，[pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md)类型。 |
| scale | 输入 | 可选，是否使能量化功能及设置量化模式下的量化参数，数据在搬出L0C时由Fixpipe乘以该比例并转换到目的数据类型。不同的传入形式会影响量化粒度，支持如下类型：<br>- **float类型**：直接传入固定值（如scale = 2.0），适用于整块tile使用同一比例。<br>- **Scalar类型**：量化比例在运行时确定，需按数据类型传值。<br>&nbsp;&nbsp;- DT_FP32：直接传原始比例值（如0.5）。<br>&nbsp;&nbsp;- DT_INT32、DT_INT64：传预编码的float32位模式转成的整数（如`struct.pack("!f", 0.5)`）。<br>- **Tile类型**：每列使用独立比例，需满足以下要求：<br>&nbsp;&nbsp;- target_memory必须为pl.MemorySpace.Scaling。<br>&nbsp;&nbsp;- shape为[1, N]（列量化），N必须是16的倍数且N ≤ 512。<br>&nbsp;&nbsp;- dtype为DT_INT64。<br>&nbsp;&nbsp;- 目的操作数的Tile数据类型为DT_INT8时，Scaling tile每个DT_INT64元素的bit46需置1，用于选择有符号量化；未置位时L0C Buffer中的负值会被按无符号解读。<br>&nbsp;&nbsp;- 用户需要先把比例数据从GM搬到L1，再搬到Scaling，并完成MTE1→FIX同步。 |
| order | 输入 | 可选，维度映射，List[int]类型，指定源Tile各维度对应的目标Tensor维度。<br>- 各维度编号必须在目标Tensor的维度范围内、不能重复。<br>- 仅支持按升序排列。<br>- 省略时对应目标Tensor的最后两个维度。GM分型为NZ时，只能指定为目标Tensor的最后两个维度。<br>- 当Tensor为1维时，不支持传入order参数。 |
| atomic | 输入 | 可选，原子写模式，[pypto_pro.language.AtomicType](../basic_data_structures/AtomicType.md)类型。 |
| phase | 输入 | 可选，分块写回阶段，[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)类型。<br>- 不支持与Tile类型的scale同时使用。 |

## 约束说明

- 数据类型和分形要求

| 源 → 目的 | 分形要求 | 数据类型要求 |
|---|---|---|
| UB → GM | 源与目的分形必须相同，支持ND、DN、NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| L0C Buffer → GM（不配置scale） | NZ → ND，NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32/DT_FP16/DT_BF16。 |
| L0C Buffer → GM（配置scale） | NZ → ND，NZ → NZ。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP8E4M3FN/DT_FP16/DT_BF16/DT_FP32，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |

- GM NZ布局：其物理排布、分形轴和完整Tensor的shape约束见[TensorLayout](../basic_data_structures/TensorLayout.md)。store还需满足以下NZ搬运约束：
    - Tile shape和valid M/N须满足M按16、N按目标Tensor dtype对应的C0对齐，N方向offset也须按C0对齐。
    - L0C Buffer中的Tile直接写回GM时，若一次写入多个N分形（valid N大于C0），写回范围须覆盖目标Tensor完整的NZ物理M轴。若部分M跨多个N分形时，需先搬到UB，再从UB写回GM。

- 接口不会自动清零目标Tensor。首次累加前，调用方必须将目标区域初始化为零或预期的累加初值。

- 多核同时累加同一目标地址时，每次更新具有原子性。由于浮点加法不满足结合律，更新顺序不同时结果可能存在微小差异。

## 返回值说明

无。

## 调用示例

### UB写回GM

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
