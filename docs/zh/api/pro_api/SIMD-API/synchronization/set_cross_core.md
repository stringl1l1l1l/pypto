# pypto_pro.language.system.set_cross_core

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

核间同步控制接口，与[pypto_pro.language.system.wait_cross_core](wait_cross_core.md)配合使用。event_id用于指定核间同步事件，每个事件ID对应一个初始值为0的计数器。pipe指定流水中的前序指令完成后，执行pypto_pro.language.system.set_cross_core使对应计数器加1。执行配对的pypto_pro.language.system.wait_cross_core时，如果计数器为0，则阻塞pipe指定流水中的后续指令；如果计数器大于0，则计数器减1，后续指令继续执行。

接口支持同类AIC或AIV全核同步、同一AI Core内的AIV间同步，以及同一AI Core内AIC与AIV之间的同步。参与同步的核和信号配对方式由sync_mode确定。

## 函数原型

```python
pypto_pro.language.system.set_cross_core(
    *,
    pipe: PipeType,
    event_id: Union[int, Scalar],
    sync_mode: CrossCoreSyncMode = pypto_pro.language.CrossCoreSyncMode.INTRA_BLOCK,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| pipe | 输入 | [pypto_pro.language.PipeType](../basic_data_structures/PipeType.md)枚举值，表示发送信号所在的硬件流水。该流水中的前序指令完成后，SET才生效。sync_mode为INTER_BLOCK、INTER_SUBBLOCK或INTRA_BLOCK时，可取M、V、MTE1、MTE2、MTE3、FIX，不支持S和ALL；sync_mode为UNICAST_BLOCK时还可取S，但仍不支持ALL。该值可以与配对的pypto_pro.language.system.wait_cross_core的pipe不同。 |
| event_id | 输入 | 核间同步事件ID。支持Python整型常量或运行时整数Scalar表达式。Python整型常量当前只能取0～15。动态表达式须由调用方保证运行时取值合法：INTER_BLOCK、INTER_SUBBLOCK、INTRA_BLOCK取0～15；UNICAST_BLOCK在AIV侧取0～15，在AIC侧取0～31。UNICAST_BLOCK中，AIV0发送的0～15与AIC等待的0～15配对，AIV1发送的0～15与AIC等待的16～31配对；AIC发送的0～15与AIV0等待的0～15配对，AIC发送的16～31与AIV1等待的0～15配对。每个事件ID对应的计数器取值范围为0～15；同一事件的信号未被WAIT消费时，连续发送超过15次SET会触发异常并中断执行。对于INTER_BLOCK、INTER_SUBBLOCK和INTRA_BLOCK，同一核复用事件ID或将同一事件ID用于不同同步模式前，必须完成该事件ID在前一同步过程中的所有SET和WAIT。与[pypto_pro.language.system.sync_all](sync_all.md#约束说明)同时使用时，须避开HARD模式占用的事件ID；使用自动流水编排时，还应避免与其分配的事件ID冲突。同一核连续发送多个SET时，不保证不同事件ID之间的生效顺序；存在先后依赖时，应先完成前一组SET/WAIT。 |
| sync_mode | 输入 | 核间同步模式，用于指定参与同步的核以及SET/WAIT信号的配对方式。须与配对的pypto_pro.language.system.wait_cross_core使用相同模式，取值参见[pypto_pro.language.CrossCoreSyncMode](../basic_data_structures/CrossCoreSyncMode.md)。 |

## 约束说明

- 必须存在与当前调用匹配的pypto_pro.language.system.wait_cross_core，并保证所有参与同步的核均能到达同步点，否则可能发生死锁。
- 使用INTER_BLOCK时，多流或多个算子并发执行，且并发算子申请的核数总和超过物理核数时，如果至少两个并发算子使用核间同步，部分核可能因未被调度而无法到达同步点，造成死锁。须保证每个同步算子所需的核能够同时执行。

## 返回值说明

无。

## 调用示例

### INTER_BLOCK

```python
with pl.section_vector():
    # 本AIV上的前置操作。
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=0,
        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
    )
```

### INTER_SUBBLOCK

```python
with pl.section_vector():
    # AIV0和AIV1各自执行前置操作。
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=1,
        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
    )
```

### INTRA_BLOCK

```python
with pl.section_vector():
    # AIV0和AIV1完成前置操作后分别发送信号。
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=2,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
    )
```

### UNICAST_BLOCK

```python
with pl.section_vector():
    if pl.get_subblock_idx() == 0:
        # 仅AIV0发送信号。
        pl.system.set_cross_core(
            pipe=pl.PipeType.S,
            event_id=15,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
```

### 完整Kernel示例

```python
import pypto_pro.language as pl


@pl.jit()
def cross_core_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ),
        addr=0x10000)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x0000)
        tile_y = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x2000)
        tile_sum = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x4000)
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                        layout=pl.NZ),
            addr=0x6000)

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                        layout=pl.NZ),
            addr=0x0000)
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                        layout=pl.NZ),
            addr=0x0000)
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                        layout=pl.ZN),
            addr=0x0000)
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                        layout=pl.NZ, fractal=1024),
            addr=0x0000)

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.move(v1_left, v1_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])
```
