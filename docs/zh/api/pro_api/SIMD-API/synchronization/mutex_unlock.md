# pypto_pro.language.system.mutex_unlock

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

在指定pipe上释放mutex_id对应的缓冲区互斥资源，使等待该资源的其他pipe能够继续执行。该接口与mutex_lock配合使用。

## 函数原型

```python
pypto_pro.language.system.mutex_unlock(
    *,
    pipe: PipeType,
    mutex_id: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| pipe | 输入 | pypto_pro.language.PipeType枚举值，必须是MTE1/MTE2/MTE3/V/M/S/FIX中的一条具体pipe；不允许PipeType.ALL。 |
| mutex_id | 输入 | Python整数、结果为整数的常量表达式，或整数类型的运行时Scalar表达式。静态ID的取值范围为[0, 31]，不接受bool；动态ID的运行时取值需在[0, 31]范围内。 |

## 约束说明

- mutex_unlock必须与此前同一pipe、同一mutex_id的mutex_lock成对使用。
- 不得遗漏mutex_unlock，也不得在未获取对应互斥资源时调用mutex_unlock。
- 嵌套、复用、控制流及自动mutex等公共约束，参见[pypto_pro.language.system.mutex_lock](mutex_lock.md#约束说明)。

## 返回值说明

无。

## 调用示例

下面的Kernel在auto_mutex=False时计算out = x + x。每个mutex_unlock均释放同一pipe上此前通过相同mutex ID获取的互斥资源。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=False)
def mutex_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x4000)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(tile_x, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=1)
        pl.add(tile_out, tile_x, tile_x)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=1)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=1)
        pl.store(out, tile_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=1)
```
