# pypto_pro.language.PipeType

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

硬件执行单元（流水）的类型枚举，标记某个操作运行在哪条流水上。是同步控制（[sync_src](../synchronization/sync_src.md)、[sync_dst](../synchronization/sync_dst.md)、[mutex_lock](../synchronization/mutex_lock.md)等）里set_pipe/wait_pipe/pipe参数的取值来源。

AI处理器内部有多条并行流水，各自负责不同阶段的数据搬运和计算。正确理解每条流水的职责是写好同步控制的基础。

## 原型定义

```python
PYPTO_DECLARE_ENUM(
    PipeType,
    MTE1,
    MTE2,
    MTE3,
    M,
    V,
    S,
    FIX,
    ALL
)
```

## 参数说明

| 参数值 | 说明 |
|---|---|
| MTE1 | 搬运流水1，负责L1 Buffer → L0A Buffer/L0B Buffer/BiasTable Buffer/L0A_MX Buffer/L0B_MX Buffer的数据搬运。 |
| MTE2 | 搬运流水2，负责GM → L1 Buffer/UB的load搬入。 |
| MTE3 | 搬运流水3，负责UB → GM的store搬出和UB → L1 Buffer的move搬运。 |
| M | 矩阵计算流水，负责Cube/MAD等matmul计算。 |
| V | 向量计算流水，负责element-wise、reduce、cast、quant、dequant等向量操作。 |
| S | 标量流水，负责getval、setval等标量操作。 |
| FIX | Fixpipe流水，负责将累加器结果从L0C Buffer搬运至GM、UB或L1 Buffer，以及随路量化/反量化等操作。 |
| ALL | 表示本AI Core的全部流水，包括V、M、MTE1、MTE2、MTE3和FIX等流水。 |

## 约束说明

### 数据搬运场景

[load](../memory_data_movement/load.md)、[move](../memory_data_movement/move.md)和[store](../memory_data_movement/store.md)的流水由源/目的内存空间自动决定：

| 操作 | 源 → 目的 | 流水 |
|---|---|---|
| load | GM → L1 Buffer/UB | MTE2 |
| store | UB → GM | MTE3 |
| store | L0C Buffer → GM | FIX |
| move | L1 Buffer → L0A Buffer/L0B Buffer/BiasTable Buffer/L0A_MX Buffer/L0B_MX Buffer | MTE1 |
| move | L1 Buffer → Fixpipe Buffer | FIX |
| move | L1 Buffer → UB | V |
| move | L0C Buffer → UB | FIX |
| move/insert | L0C Buffer → L1 Buffer | FIX |
| move | UB → UB | V |
| move | UB → L1 Buffer | MTE3 |
| move | 其余 | V |

### 流水同步场景

典型同步模式：

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load后V才能计算 | MTE2 | V | 确保GM→UB搬运完成 |
| 计算后MTE3才能store | V | MTE3 | 确保向量计算完成 |
| store后MTE2才能load | MTE3 | MTE2 | 确保UB→GM搬出完成再搬入新数据 |
| L1 Buffer→L0A Buffer搬运后M才能计算 | MTE1 | M | 确保矩阵操作数就位 |
| matmul后FIX才能读出结果 | M | FIX | 确保矩阵计算完成 |
| 标量读写 | MTE2/MTE3 | S | getval/setval走标量流水 |

## 调用示例

### 流水同步

```python
import pypto_pro.language as pl
# load 后同步：MTE2 置位，V 等待
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

# 计算后同步：V 置位，MTE3 等待
pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
```
