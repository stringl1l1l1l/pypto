# pypto_pro.language.system.set_mm_layout_transform

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

用于matmul计算中切换Fixpipe结果读出方向。开启后，Fixpipe沿N方向从L0C Buffer读取数据。

## 函数原型

```python
pypto_pro.language.system.set_mm_layout_transform(*, enabled: bool) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| enabled | 输入 | Fixpipe结果读出方向切换开关，bool类型，必须在编译期确定。True表示沿N方向读取，False表示沿M方向读取。仅在matmul计算与Fixpipe需要并行访问同一块L0C Buffer时使用。 |

## 约束说明

- 仅当matmul计算与Fixpipe并行访问同一块L0C Buffer时使用；完成结果搬出后将enabled设为False。两者串行执行或使用不同L0C Buffer时无需调用。

## 返回值说明

无。

## 调用示例

### K维分块累加时切换Fixpipe结果读出方向

```python
import pypto_pro.language as pl

TILE_ACC = 128
K_SIZE_ACC = 256


@pl.jit(auto_mutex=True)
def mm_layout_kernel(
    a: pl.Tensor[[TILE_ACC, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE_ACC], pl.DT_FP16],
    c: pl.Tensor[[TILE_ACC, TILE_ACC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        # 当前流水调度要求Fixpipe沿N方向读取结果。
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE_ACC):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```
