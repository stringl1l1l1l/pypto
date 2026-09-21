# pypto_pro.language.section_vector

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

标识Kernel中的Vector执行域，需与with语句配合使用。Vector执行域用于组织GM与UB之间的数据搬运以及UB Tile上的向量计算，包括逐元素运算、归约和转置等。

同一Kernel中可先后定义多个Vector执行域，也可与[pypto_pro.language.section_cube](section_cube.md)标识的Cube执行域配合使用。

## 函数原型

```python
pypto_pro.language.section_vector() -> ContextManager
```

## 参数说明

无。

## 约束说明

- section_vector不能与section_vector或section_cube嵌套使用。
- 所有执行Tile操作的API必须位于section_vector或section_cube内部，不能直接出现在Kernel函数体顶层。
- pipeline模式下，stage调用链须严格交替Cube/Vector，不支持两个连续同类型区域中的stage。

## 返回值说明

返回一个上下文管理器，用于界定Vector执行域。

## 调用示例

### Vector区域向量计算

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_vector_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
