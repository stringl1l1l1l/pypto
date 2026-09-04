# pypto_pro.language.set_validshape

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

设置Tile或TileGroup的有效shape范围，用于处理尾块或非满Tile的场景。传入单个Tile时，直接设置该Tile的有效数据范围；传入TileGroup时，对group中所有Tile批量设置相同的valid_shape，适用于全局只需设置一次、后续直接next()的场景。pypto_pro.language.make_tile与[pypto_pro.language.TileType](../../../basic_data_structures/TileType.md)的valid_shape后端缺省行为等同于[-1, -1]（动态模式），一般无需显式指定。

## 函数原型

```python
pypto_pro.language.set_validshape(tile: Union[Tile, TileGroup], shape: List[int]) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输入 | 目标Tile或pypto_pro.language.make_tile_group返回的TileGroup。Tile数据类型支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。valid_shape后端缺省值为[-1, -1]，无需显式指定。 |
| shape | 输入 | 长度为2的有效shape序列，两个元素均为整型常量或运行时整型标量表达式（支持循环变量），元素须为正整数，且分别不超过Tile shape对应维度。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def validshape_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                            valid_shape=[-1, -1])
    tile_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile = tile_group.current()
        pl.set_validshape(tile, [rows, cols])
        pl.load(tile, a, [0, 0])
        pl.store(out, tile, [0, 0])
```

### 其他典型用法

```python
# matmul 尾块处理
pl.set_validshape(q_mat_buf[q_count % 2], [TD, actual_sq])

# TileGroup访问器和下标都直接返回Tile
pl.set_validshape(a_wide_group.current(), [256, 64])
pl.set_validshape(a_wide_group[0], [256, 64])
```

### TileGroup批量设置

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tile_group_validshape_kernel(
    a: pl.Tensor[[128, 128], pl.DT_FP16],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    output: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
    tile_a = pl.make_tile_group(
        type=tile_type, addrs=0x0000, mutex_ids=[0])
    pl.set_validshape(tile_a, [rows, cols])
    with pl.section_vector():
        cur_a = tile_a.current()
        pl.load(cur_a, a, [0, 0])
        pl.store(output, cur_a, [0, 0])
```
