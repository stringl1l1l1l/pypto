# pypto_pro.language.store_tile

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

把UB或L0C Buffer中的Tile结果写回GM。与[pypto_pro.language.store](store.md)的区别在于，偏移以Tile块索引为单位，内部自动按块索引乘以Tile形状，换算成绝对元素坐标。该接口是[pypto_pro.language.load_tile](load_tile.md)的反向操作。

例如Tile形状为[64, 128]时，tile_offsets=[2, 2]等价于[pypto_pro.language.store](store.md)的绝对偏移[128, 256]。

下图以UB源Tile为例展示按块索引写回GM Tensor的过程。块索引先换算为元素偏移，再确定目标块的落点；L0C Buffer中的源Tile通过Fixpipe写回。

![store_tile按块索引把Tile写回GM](../../figures/store_tile_block_offset.jpg "store_tile按块索引把Tile写回GM")

## 函数原型

```python
pypto_pro.language.store_tile(
    dst_tensor: Tensor,
    src_tile: Tile,
    tile_offsets: Offset,
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
| dst_tensor | 输出 | 目的操作数，Tensor类型，存储空间为GM。支持的数据类型和分形组合详见[约束说明](#约束说明)。写入范围不得越过Tensor边界。 |
| src_tile | 输入 | 源操作数，Tile类型，存储空间为UB或L0C Buffer。位于UB时通过MTE3流水写回，首地址按32字节对齐；位于L0C Buffer时通过Fixpipe写回，首地址按64字节对齐。 |
| tile_offsets | 输入 | 目标Tensor的Tile块偏移，List[int或Scalar]类型。由order指定的维度按块索引乘以Tile对应维度大小换算，其余维度按绝对偏移使用；不支持负数索引，换算后的绝对偏移不得超过对应维度的形状。 |
| relu_pre_mode | 输入 | 预处理模式，pypto_pro.language.ReluPreMode类型，可选。支持ReluPreMode.NormalRelu。 |
| scale | 输入 | 量化比例，float、Scalar或Tile类型，可选。float或运行时Scalar表示整块Tile使用同一比例；运行时DT_FP32 Scalar直接传比例值，运行时DT_INT32或DT_INT64 Scalar须传入预编码的float32位模式。Tile表示逐列量化，须位于Fixpipe Buffer，数据类型为DT_INT64，形状为[1, N]，其中N为16的倍数且不大于512。 |
| order | 输入 | 维度映射，List[int]类型，可选。指定源Tile的两个维度分别对应目标Tensor的哪两个维度，仅支持包含两个升序维度索引的列表，例如[0, 2]；维度索引必须在目标Tensor的维度范围内。省略时使用目标Tensor的最后两个维度。 |
| atomic | 输入 | 原子写模式，pypto_pro.language.AtomicType类型，可选。支持AtomicType.AtomicNone（覆盖写）和AtomicType.AtomicAdd（原子累加）。 |
| phase | 输入 | 分块写回阶段，[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)类型，可选。支持STPhase.Partial和STPhase.Final；scale为逐列量化Tile时不能同时设置该参数。 |

## 约束说明

### 数据类型和分形要求（与[store](store.md#约束说明)一致）

| 源 → 目的 | 分形要求 | 数据类型要求 |
|---|---|---|
| UB → GM | 源与目的分形必须相同，支持ND、DN、NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| L0C Buffer → GM（不配置scale） | NZ → ND，NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32/DT_FP16/DT_BF16。 |
| L0C Buffer → GM（配置scale） | NZ → ND，NZ → NZ。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP8E4M3FN/DT_FP16/DT_BF16/DT_FP32，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |

当dst_tensor声明为NZ时，其物理排布和完整Tensor shape约束见[TensorLayout](../basic_data_structures/TensorLayout.md)，同布局搬运、源Tile、order和L0C Buffer直接写回约束与[store](store.md#约束说明)一致。store_tile还需满足以下NZ搬运约束：

- tile_offsets按Tile块索引寻址：最后两项分别乘以Tile的M、N shape，前导项选择batch；换算后的M、N offset需分别按16和目标Tensor dtype对应的C0对齐。

## 返回值说明

无。

## 调用示例

### 按Tile块索引从UB写回GM

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def store_tile_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP16],   # 4 个 64x64 的块
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        cur_x = tile_x.current()
        pl.load(cur_x, x, [0, 0])
        for ti in pl.range(0, 4, 1):
            pl.store_tile(out, cur_x, [ti, 0])
```

### 高维Tensor写回

```python
# 四维BSND Tensor：Tile对应第1、3维，其余维度使用绝对偏移
pl.store_tile(p_buf, p_f16, [b_idx, qi * 2 + sub_id, n_idx, ki], order=[1, 3])
```
