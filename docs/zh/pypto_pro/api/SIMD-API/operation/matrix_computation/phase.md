# phase使用约束（AccPhase / STPhase）

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

`matmul` / `matmul_acc` / `matmul_mx` / `matmul_mx_acc`的`phase`参数（`pl.AccPhase`）与`store` / `store_tile` / `move`的`phase`参数（`pl.STPhase`）共同控制Cube（矩阵乘）与（L0C→GM/L0C→UB搬运）之间的**unit_flag硬件握手**。正确使用phase可以省去软件同步、提升流水并行度；使用不当则会导致精度问题或设备卡死。

## 硬件unit_flag机制

### matmul系列接口（AccPhase）

`phase=pl.AccPhase.Partial`或`phase=pl.AccPhase.Final`均会使能硬件的unitFlag功能：

- **unit_flag = 0**：硬件直接写入Acc（L0C）。
- **unit_flag = 1**：硬件写入Acc的操作会被暂停，直到unit_flag被设置回0。

两者的区别：

| 模式 | 检查unit_flag | 设置unit_flag |
|---|---|---|
| `Partial` | 是（等待unit_flag = 0才写入） | 否（不改变unit_flag） |
| `Final` | 是（等待unit_flag = 0才写入） | 是（写入后将unit_flag置为1） |

### store / store_tile / move（STPhase）

`phase=pl.STPhase.Partial`或`phase=pl.STPhase.Final`均会使能硬件的unitFlag功能：

- **unit_flag = 1**：硬件直接读Acc（L0C）。
- **unit_flag = 0**：硬件读Acc的操作会被暂停，直到unit_flag被设置为1。

两者的区别：

| 模式 | 检查unit_flag | 设置unit_flag |
|---|---|---|
| `Partial` | 是（等待unit_flag = 1才读取） | 否（不改变unit_flag） |
| `Final` | 是（等待unit_flag = 1才读取） | 是（读取后将unit_flag置为0） |

## phase与自动同步的关系

| 配置 | 自动同步 | 同步机制 |
|---|---|---|
| 配置了phase |**不自动插入同步** | 靠硬件unit_flag实现Matmul（M流水）与FixPipe之间的同步 |
| 未配置phase |**自动插入同步** | 框架自动插入M流水与FixPipe流水的软件同步 |

## 使用约束

如果phase使用不当，可能会导致精度问题或者卡死现象。使用时必须保证：

1. **配对使用**：如果任一`matmul`系列接口使用了phase，对应的`store`或`store_tile`也需要使用phase。
2. **Final收尾**：对于同一块L0C，`matmul`系列接口的最后一轮写操作，以及`store`或`store_tile`的最后一轮读操作，必须使用`Final`模式。

## 错误案例

### 错误案例一：matmul无Final导致卡死

```python
pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

**现象**：卡死。

**原因**：`matmul`使用`Partial`只检查unit_flag不会设置unit_flag，unit_flag始终为0。`store`使用`Final`等待unit_flag被设置成1才能读取，但unit_flag永远不会被置1，FixPipe一直等待 → 卡死。

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

- **软件同步角度**：`store`未配置phase，框架会自动插入FixPipe流水同步；但`matmul`配置了phase，不会自动插入M流水同步。两种同步机制不匹配。
- **硬件unit_flag角度**：`store`未配置phase，不受硬件unit_flag值影响，FixPipe不会等待unit_flag。

上述两种情况，FixPipe搬运L0C数据都不会严格等待Matmul计算完成，导致读到未完成的数据。

### 错误案例三：循环内store(Final)后matmul卡死

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
    pl.store(out, ac, [0, 0], phase=pl.AccPhase.Partial)
```

**现象**：卡死。

**原因**：

- 第一轮循环：`matmul(Final)`将unit_flag设置成1，`store(Partial)`能将L0C数据搬运出去，但未改变unit_flag的值（仍为1）。
- 第二轮循环：由于共用同一块L0C内存，`matmul`等待unit_flag变更为0，但unit_flag始终为1 → 卡死。

## 正确用法示例

### 单次matmul（无K维累加）

不传phase，框架自动插入同步：

```python
pl.matmul(ac, al, br)
pl.store(out, ac, [0, 0])
```

### K维分块累加（多块）

首块`Partial`，中间块`Partial`，末块`Final`，store用`STPhase.Final`：

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

### Flash Attention

matmul 结果需要 vector 核做后处理（如 softmax）时，必须通过 `move` 将累加器数据搬到 UB，`store` 只能直接写 GM，无法在 UB 上做后续计算。以 Flash Attention 的 QK matmul 为例：

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
