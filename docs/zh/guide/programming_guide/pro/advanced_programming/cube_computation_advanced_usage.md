# Cube计算进阶

常规Cube计算由框架自动处理流水同步。对于K维分块累加、L0C Buffer结果分阶段搬出等需要精细控制Cube计算流水与Fixpipe搬运流水的场景，可以通过`phase`参数使用硬件`unit_flag`完成同步。本节介绍这一进阶机制的适用场景、支持接口、同步语义、使用约束和代码示例。

## Phase使用场景

在Matmul计算过程中，如果Cube计算流水将结果写入L0C Buffer，随后需要通过Fixpipe将L0C Buffer中的数据搬运至GM、UB或L1 Buffer，可以通过配置AccPhase和STPhase，使用硬件unit_flag完成Cube与Fixpipe之间的同步。

相比框架默认插入的软件同步，配置`phase`可以减少Matmul计算流水与Fixpipe在L0C Buffer上的软件同步开销，提高流水并行度。

典型使用场景包括：

- K维分块Matmul，多次计算结果累加至同一块L0C Buffer后再搬出。
- Matmul结果需要搬运至UB，由Vector核继续进行后处理，例如Flash Attention中的Softmax计算。
- 其他需要在Cube计算流水与Fixpipe搬运流水之间使用硬件unit_flag完成同步的场景。

对于单次Matmul计算，如果没有使用硬件`unit_flag`进行流水同步的需求，可以不配置`phase`，由框架自动插入软件同步。

## Phase机制

Matmul系列接口的`phase`参数使用`pypto_pro.language.AccPhase`，Fixpipe搬出接口的`phase`参数使用`pypto_pro.language.STPhase`。

两者共同控制Cube计算流水与Fixpipe搬运流水之间的硬件unit_flag握手。

相关枚举说明请参见：

- [`pypto_pro.language.AccPhase`](../../../../api/pro_api/SIMD-API/basic_data_structures/AccPhase.md)
- [`pypto_pro.language.STPhase`](../../../../api/pro_api/SIMD-API/basic_data_structures/STPhase.md)

支持配置AccPhase的接口包括：

- [`pypto_pro.language.matmul`](../../../../api/pro_api/SIMD-API/cube_computation/matmul.md)
- [`pypto_pro.language.matmul_acc`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_acc.md)
- [`pypto_pro.language.matmul_mx`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx.md)
- [`pypto_pro.language.matmul_mx_acc`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx_acc.md)

支持配置STPhase的接口包括：

- [`pypto_pro.language.store`](../../../../api/pro_api/SIMD-API/memory_data_movement/store.md)
- [`pypto_pro.language.store_tile`](../../../../api/pro_api/SIMD-API/memory_data_movement/store_tile.md)
- [`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md)
- [`pypto_pro.language.insert`](../../../../api/pro_api/SIMD-API/memory_data_movement/insert.md)

### AccPhase

当Matmul系列接口配置pypto_pro.language.AccPhase.Partial或pypto_pro.language.AccPhase.Final时，会使能硬件unit_flag功能。

对于Cube向L0C Buffer写入数据：

- unit_flag = 0：允许硬件写入L0C Buffer。
- unit_flag = 1：写入操作暂停，直到unit_flag重新变为0。

Partial和Final的行为如下。

| 模式 | 检查unit_flag | 设置unit_flag |
| --- | --- | --- |
| Partial | 是，等待unit_flag = 0后写入 | 否，不改变unit_flag |
| Final | 是，等待unit_flag = 0后写入 | 是，写入完成后将unit_flag置为1 |

因此，在多次Matmul累加场景中，中间计算通常使用Partial，最后一次写入L0C Buffer的计算使用Final，通知Fixpipe可以开始读取数据。

### STPhase

当Fixpipe搬出接口配置pypto_pro.language.STPhase.Partial或pypto_pro.language.STPhase.Final时，同样会使能硬件unit_flag功能。

对于Fixpipe从L0C Buffer读取数据：

- unit_flag = 1：允许硬件读取L0C Buffer。
- unit_flag = 0：读取操作暂停，直到unit_flag变为1。

Partial和Final的行为如下。

| 模式 | 检查unit_flag | 设置unit_flag |
| --- | --- | --- |
| Partial | 是，等待unit_flag = 1后读取 | 否，不改变unit_flag |
| Final | 是，等待unit_flag = 1后读取 | 是，读取完成后将unit_flag置为0 |

STPhase.Final读取完成后将unit_flag恢复为0，使同一块L0C Buffer能够继续被后续Matmul计算使用。

### Phase与自动同步的关系

是否配置`phase`会影响框架在Matmul计算流水与Fixpipe之间使用的同步方式。

| 配置 | L0C Buffer同步 | 同步机制 |
| --- | --- | --- |
| 配置`phase` | 不插入Matmul计算流水与Fixpipe之间的软件同步 | 通过硬件unit_flag完成Matmul计算流水与Fixpipe之间的同步；L1 Buffer、L0A Buffer和L0B Buffer等其他Tile的自动同步不受影响 |
| 未配置`phase` | 插入Matmul计算流水与Fixpipe之间的软件同步 | 框架通过软件同步保证Matmul计算流水与Fixpipe的执行顺序 |

配置`phase`只影响Matmul计算流水与Fixpipe之间针对L0C Buffer的同步，不影响其他Buffer的自动同步机制。

## Phase使用约束

`phase`配置不当可能导致计算结果错误或设备卡死。使用时需要满足以下约束：

1. **AccPhase和STPhase需要配合使用。**

   如果Matmul系列接口配置了phase，对应从该L0C Buffer读取数据的store、store_tile、move或insert也需要正确配置phase。

2. **最后一次写入和读取需要使用Final模式。**

   对于同一块L0C Buffer，Matmul系列接口最后一次写操作需要使用AccPhase.Final，对应Fixpipe最后一次读操作需要使用STPhase.Final。

## Phase使用示例

### 单次matmul（无K维累加）

不传phase，框架自动插入同步：

```python
pl.matmul(ac, al, br)
pl.store(out, ac, [0, 0])
```

### K维分块累加（多块）

首块Partial，中间块Partial，末块Final，store用STPhase.Final：

```python
with pl.section_cube():
    ac = acc.current()
    for k in pl.range(0, K_TOTAL, TILE_K):
        ...
        if k == 0:
            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)        # 首块
        elif k < K_TOTAL - TILE_K:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial) # 中间块
        else:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)   # 末块
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)                # Final 收尾
```

### Flash Attention示例

matmul 结果需要 vector 核做后处理（如 softmax）时，必须通过 move 将累加器数据搬到 UB，store 只能直接写 GM，无法在 UB 上做后续计算。以 Flash Attention 的 QK matmul 为例：

```python
with pl.section_cube():
    ac = acc.current()
    for k in pl.range(0, K_TOTAL, TILE_K):
        ...
        if k == 0:
            pl.matmul(ac, q, k, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_acc(ac, ac, q, k, phase=pl.AccPhase.Final)
    # 搬到 UB 供 vector 核做 softmax（store 做不到 L0C→UB）
    pl.move(qk_vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN,
            phase=pl.STPhase.Final)
with pl.section_vector():
    # softmax: row max → sub → exp → sum → scale
    pl.maximum(reduce_max, qk_vec, tmp, dim=1)
    pl.expand_sub(tmp, qk_vec, reduce_max, dim=1)
    # ... exp / sum / scale ...
    pl.store(p_buf, qk_vec, [...])
```

## Phase常见错误

### 错误案例一：matmul无Final导致卡死

```python
pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

**现象**：卡死。

**原因**：matmul使用Partial只检查unit_flag不会设置unit_flag，unit_flag始终为0。store使用Final等待unit_flag被设置成1才能读取，但unit_flag永远不会被置1，Fixpipe一直等待 → 卡死。

### 错误案例二：store未配置phase导致精度问题

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    if ki == 0:
        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
    else:
        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
pl.store(out, ac, [0, 0])
```

**现象**：精度问题。

**原因**：

- **软件同步角度**：store未配置phase，框架会自动插入Fixpipe同步；但matmul配置了phase，不会自动插入matmul计算流水同步。两种同步机制不匹配。
- **硬件unit_flag角度**：store未配置phase，不受硬件unit_flag值影响，Fixpipe不会等待unit_flag。

上述两种情况，Fixpipe搬运L0C数据都不会严格等待Matmul计算完成，导致读到未完成的数据。

### 错误案例三：循环内store(Final)后matmul卡死

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Partial)
```

**现象**：卡死。

**原因**：

- 第一轮循环：matmul(Final)将unit_flag设置成1，store(Partial)能将L0C数据搬运出去，但未改变unit_flag的值（仍为1）。
- 第二轮循环：由于共用同一块L0C内存，matmul等待unit_flag变更为0，但unit_flag始终为1 → 卡死。
