# pypto_pro.language.system.mutex_lock

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

在指定pipe上获取mutex_id对应的缓冲区互斥资源。资源尚未释放时，当前pipe等待，直至能够获取该资源。该接口用于防止多条pipe同时访问同一片上缓冲区。

## 函数原型

```python
pypto_pro.language.system.mutex_lock(
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

- mutex_lock必须与同一pipe、同一mutex_id的mutex_unlock成对使用，且先调用mutex_lock，再调用mutex_unlock。
- 同一pipe上，在前一次mutex_lock尚未由对应的mutex_unlock释放时，不得再次获取同一mutex_id，否则第二次获取会持续等待并导致死锁。手动和自动生成的mutex操作也不得在同一pipe上重复获取尚未释放的同一ID。
- 同一mutex_id对应的mutex_lock和mutex_unlock不得嵌套使用，无论各组操作的pipe是否相同。使用自动mutex时，也须避免与显式mutex操作形成同一ID的嵌套。
- 同一pipe上连续使用相同mutex_id的多组mutex_lock和mutex_unlock，不能保证该pipe中各组操作依次完成。需要保证同一流水中的前一操作完成后再执行后一操作时，应优先使用对应的单流水屏障接口；当前流水没有对应的单流水屏障接口时，可使用[pypto_pro.language.system.bar_all](bar_all.md)。
- mutex_lock和mutex_unlock需要位于对称的控制流路径中，确保每次获取的互斥资源均会被释放。
- auto_mutex=True仅对带mutex元数据的Tile自动生成互斥操作；显式调用的mutex_lock仍会保留，自动同步和手动同步可以在同一Kernel中使用。
- 常规单缓冲、双缓冲和N缓冲场景推荐使用[make_tile_group](../resource_management/make_tile_group.md)配合auto_mutex=True。需要精确控制加锁pipe和插入位置时，再使用mutex_lock和mutex_unlock。

## 返回值说明

无。

## 调用示例

下面的Kernel在auto_mutex=False时计算out = x + x。输入UB使用mutex ID 0约束MTE2和V的访问顺序，输出UB使用mutex ID 1约束V和MTE3的访问顺序。每次mutex_lock之后均在同一pipe上调用对应的mutex_unlock。

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
