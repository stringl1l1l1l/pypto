# pypto_pro.language.system.bar_all

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

等待当前AI Core内V、M、MTE1、MTE2、MTE3和FIX等流水中此前下发的操作完成。

## 函数原型

```python
pypto_pro.language.system.bar_all() -> None
```

## 参数说明

无。

## 约束说明

- 支持在Cube区段或Vector区段中调用。
- 仅同步当前AI Core内的全部流水，不是多个AI Core之间的全局屏障。多核同步请使用[pypto_pro.language.system.sync_all](sync_all.md)。
- 本接口会等待当前AI Core内全部流水此前下发的操作完成，可能影响性能。只需要等待一条流水时，应使用对应的单流水屏障接口。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit()
def bar_all_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x2000)
    with pl.section_vector():
        for i in pl.range(0, 128, 64):
            pl.system.bar_all()
            pl.load(tile_x, x, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])
```
