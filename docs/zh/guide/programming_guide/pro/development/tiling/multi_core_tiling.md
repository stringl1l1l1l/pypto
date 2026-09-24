# 多核Tiling切分

多核Tiling切分将大Batch等大规模数据划分为较小的Tile块，再把这些Tile块分配给多个AI Core并行处理。本文介绍核间分配的方法；Tiling的概念、层次和参数来源参见[Tiling概述](tiling_overview.md)。

## 多核切分的基本写法

以矩阵乘为例，输出被划分为`m_tiles * n_tiles`个Tile，每个Tile的计算相互独立。Tile数按向上取整计算，保证最后不足一个Tile的数据也被覆盖：

```python
m_tiles = (m + tile_m - 1) // tile_m
n_tiles = (n + tile_n - 1) // tile_n
total_tiles = m_tiles * n_tiles
```

把这些Tile按行优先统一编号成`total_tiles`个任务，每个核负责其中连续的一段，再由任务号换算回二维下标：

```python
import pypto_pro.language as pl


core_id = pl.get_block_idx() // pl.get_subblock_num()
core_num = pl.get_block_num()

tiles_per_core = (total_tiles + core_num - 1) // core_num
start = core_id * tiles_per_core
end = pl.min(start + tiles_per_core, total_tiles)

for task_idx in pl.range(start, end, 1):
    tile_m_idx = task_idx // n_tiles
    tile_n_idx = task_idx % n_tiles
    # 计算第 (tile_m_idx, tile_n_idx) 个输出Tile
    ...
```

[`pypto_pro.language.get_block_num()`](../../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md)返回本次实际生效的核数，计算每个核的任务范围时应使用它的返回值。`block_dim`的默认值、调用形式和核数限制参见[Kernel核函数](../kernel_function.md#blockdim的含义与设置)。

`core_id`由[`pypto_pro.language.get_block_idx()`](../../../../../api/pro_api/SIMD-API/system_variables/get_block_idx.md)除以[`pypto_pro.language.get_subblock_num()`](../../../../../api/pro_api/SIMD-API/system_variables/get_subblock_num.md)得到。纯Vector和纯Cube Kernel中`get_subblock_num()`返回1，该除法不改变结果；MIX Kernel中Vector侧的`get_block_idx()`是Vector核的全局索引，除以每个AI Core内的Vector核数后，与Cube侧得到同一个AI Core编号，两侧才能按同一套切分处理同一批数据。

**图1 任务编号与连续分配**

![任务编号与连续分配](../../../../figures/pro/pro_multicore_contiguous_grid.png "任务编号与连续分配")

这种分配不要求任务数被核数整除：`tiles_per_core`向上取整，末尾的核用[`pypto_pro.language.min()`](../../../../../api/pro_api/Utils-API/python_syntax_sugar/min.md)把上界限制在`total_tiles`，剩余任务自然落在最后一个进入循环的核上，不需要为“除不尽”单独编写分支。任务总数少于核数时，编号超出范围的核得到空区间，不会进入循环。

算子的执行时间取决于任务最多的那个核，为`ceil(total_tiles / core_num)`个任务的耗时。

上面的编号方式是一种通用选择。算子完全可以按自身的数据复用和依赖关系定义别的切分方式，例如输出Tile之间存在跨迭代累加时，被累加的那一维必须完整留在同一个核内。

## MIX Kernel的切分

Device上有多个AI Core，每个AI Core内含Cube核（AIC）和Vector核（AIV）；在AIC与AIV为1:2的芯片上，一个AI Core内有1个Cube核和2个Vector核。同时包含Cube和Vector[执行域](../kernel_function.md#定义执行域)的Kernel称为MIX Kernel。

MIX Kernel以AI Core为启动单位，核间切分也只在AI Core这一层做一次：`get_block_num()`在两侧都返回AI Core数，`core_id`在两侧都取`get_block_idx() // get_subblock_num()`（AIC上`get_subblock_num()`返回1，AIV上返回2，两侧因此得到同一个AI Core编号）。上一节的任务划分直接沿用，Cube和Vector在同一个任务循环里按同一个`task_idx`推进：

```python
import pypto_pro.language as pl


core_id = pl.get_block_idx() // pl.get_subblock_num()
core_num = pl.get_block_num()

tiles_per_core = (total_tiles + core_num - 1) // core_num
start = core_id * tiles_per_core
end = pl.min(start + tiles_per_core, total_tiles)

for task_idx in pl.range(start, end, 1):
    with pl.section_cube():
        # 计算本任务的整块输出
        ...

    with pl.section_vector():
        # 处理Cube刚算出的这块数据
        ...
```

在1:2的芯片上，一个AI Core内有2个Vector核，两者要分摊同一个任务里Cube产出的这份数据，因此每个Vector核处理的数据量是Cube的一半。这一步用[`pypto_pro.language.get_subblock_idx()`](../../../../../api/pro_api/SIMD-API/system_variables/get_subblock_idx.md)在某一维上把数据切成两半，属于核内切分，不参与核间任务的分配：

```python
import pypto_pro.language as pl

sub_id = pl.get_subblock_idx()

first_half = (rows + 1) // 2
half_rows = first_half
if sub_id == 1:
    half_rows = rows - first_half

for task_idx in pl.range(start, end, 1):
    with pl.section_cube():
        # 计算本任务的rows行输出
        ...

    with pl.section_vector():
        if half_rows > 0:
            # 起始行 sub_id * first_half
            # 行数 half_rows
            ...
```

切分的维度由算子自身决定，行、列或batch方向都可以，前提是两个Vector核的工作互不重叠且合起来覆盖Cube的全部输出。

## 负载均衡与切分粒度

上面的分配方式保证了各核**Tile块数**接近，但没有保证各核**计算量**接近。每个Tile块的计算量相近时，两者等价；计算量差异较大时（例如因果掩码下不同行块的有效长度不同），需要额外调整：

- 调整Tile块的编号顺序，使计算量大的Tile块分散到不同的核，而不是集中在少数核上。
- 拆分计算量过大的Tile块，减少单个Tile块造成的尾部等待。

切分粒度的选择需要同时考虑两端：

- **Tile块总数要足以覆盖核数**。Tile块数少于核数时部分核不会进入循环；Tile块数接近核数时，零头会明显放大为负载差异。
- **每核分配的Tile块不宜过多**。在数据规模一定时，每核处理过多Tile块通常意味着使用核数偏少，串行迭代过长，难以充分发挥多核算力。
- **单个Tile块不宜过小**。Tile块越小，循环控制、数据搬运启动和同步的开销占比越高，且计算指令占比越小，算力不能充分发挥。

设计切分方式时重点检查：

- Tile块总数是否足以覆盖计划使用的核。
- 各核分到的Tile块数和计算量是否接近，单核是否承担了过多Tile块。
- 切分后的Tile形状是否满足数据搬运和计算接口的约束。

Tile大小、Buffer分配和Tile计算分别参考[Tile创建和操作](../tile_creation_and_operations.md)和[Tile计算](../vector_computation/tile_computation.md)。Host侧生成的运行时Tiling参数参考[Tiling参数定义与传递](tiling_parameter_definition.md)。

## 尾块处理

尾块场景需要向上取整计算Tile数以覆盖全部任务，并根据数据搬运和计算接口的对齐要求选择Tile尺寸；有效形状、填充和计算方法参考[Tile计算中的尾块处理](../vector_computation/tile_computation.md#尾块处理)。
