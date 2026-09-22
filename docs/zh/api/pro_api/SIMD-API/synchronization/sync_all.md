# pypto_pro.language.system.sync_all

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

在多个AIV核、多个AIC核，或AIV与AIC核之间建立核间屏障。参与同步的核到达sync_all后等待，直到本轮所有参与核均已到达，再继续执行。

使用FFTS硬件同步。

## 函数原型

```python
pypto_pro.language.system.sync_all(
    *,
    core_type: SyncCoreType = pypto_pro.language.SyncCoreType.MIX,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| core_type | 输入 | 可选，[pypto_pro.language.SyncCoreType](../basic_data_structures/SyncCoreType.md)枚举值，指定参与屏障的核类型，默认为pypto_pro.language.SyncCoreType.MIX。该参数不指定参与核数量。 |

## 约束说明

- 所有参与同步的核必须以相同顺序执行相同次数的sync_all。若循环次数或分支条件不一致，导致部分核少执行或多执行sync_all，可能发生死锁。MIX模式下，AIC侧与AIV侧的调用必须一一对应。
- 纯Vector Kernel使用AIV_ONLY，纯Cube Kernel使用AIC_ONLY。MIX模式要求Cube侧AIC和Vector侧AIV都执行对应的sync_all；只在一侧调用会使另一侧无法到达屏障，导致Kernel超时。
- 多流或多个算子并发执行，且并发算子申请的核数总和超过物理核数时，如果至少两个并发算子使用sync_all，部分核可能因未被调度而无法到达屏障，造成死锁。须保证每个同步算子所需的核能够同时执行。
- sync_all建立参与核之间的屏障。屏障前后需要跨核读写GM数据时，还需满足相应的数据可见性要求。
- 与set_cross_core/wait_cross_core并用时，两侧的MIX屏障必须位于该SET/WAIT对的同一侧；禁止Cube侧先执行sync_all再SET、Vector侧先WAIT再执行sync_all，否则会形成环形等待。
- sync_all会占用核间同步事件ID：AIV_ONLY在AIV侧占用14，AIC_ONLY在AIC侧占用11；MIX在AIC侧占用11～13、在AIV侧占用12～13，MIX 1:2场景的AIC还会占用28和29。与set_cross_core/wait_cross_core同时使用时，不得将这些事件ID用于尚未完成的手工核间同步。

## 返回值说明

无。

## 调用示例

下面是纯Vector Kernel展示AIV_ONLY屏障的放置方式。各AIV按核号处理互不重叠的行，TileGroup和auto_mutex负责核内pipe依赖；sync_all位于循环外，使所有参与AIV在本阶段结束后再越过屏障。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP32],
    out: pl.Tensor[[2048, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    input_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    output_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile_x = input_tiles.next()
            tile_out = output_tiles.next()
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        pl.system.sync_all(
            core_type=pl.SyncCoreType.AIV_ONLY,
        )
```

该示例使用AIV_ONLY屏障。

### 核类型

三种core_type调用方式如下。以下片段分别用于对应类型的Kernel。

```python
# 纯Vector Kernel：所有参与AIV都执行
with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        core_type=pl.SyncCoreType.AIV_ONLY,
    )

# 纯Cube Kernel：所有参与AIC都执行
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        core_type=pl.SyncCoreType.AIC_ONLY,
    )

# Cube、Vector共存的Kernel：AIC和AIV必须到达同一个MIX屏障
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        core_type=pl.SyncCoreType.MIX,
    )

with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        core_type=pl.SyncCoreType.MIX,
    )
```
