# pypto_pro.language.section_cube

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

标识Kernel中的Cube执行域，需与with语句配合使用。Cube执行域用于组织矩阵计算及相关的数据搬运操作，包括GM与L1 Buffer之间、L1 Buffer与L0A Buffer或L0B Buffer之间的数据搬运、矩阵计算以及计算结果从L0C Buffer搬出。

同一Kernel中可先后定义多个Cube执行域，也可与[pypto_pro.language.section_vector](section_vector.md)标识的Vector执行域配合使用。

## 函数原型

```python
pypto_pro.language.section_cube() -> ContextManager
```

## 参数说明

无。

## 约束说明

- section_cube不能与section_cube或section_vector嵌套使用。
- 所有执行Tile操作的API必须位于section_vector或section_cube内部，不能直接出现在Kernel函数体顶层。
- pipeline模式下，stage调用链须严格交替Cube/Vector，不支持两个连续同类型区域中的stage。

## 返回值说明

返回一个上下文管理器，用于界定Cube执行域。

## 调用示例

### Cube区域矩阵计算

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_cube_fp16_kernel(
    x: pl.Tensor[[64, 32], pl.DT_FP16],
    y: pl.Tensor[[32, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])
```
