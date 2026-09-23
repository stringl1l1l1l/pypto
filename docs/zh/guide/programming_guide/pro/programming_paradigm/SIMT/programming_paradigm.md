# 编程范式

SIMT（Single Instruction Multiple Threads，单指令多线程）是一种线程并行模型，以Thread为基本执行单元。同一份程序由多个Thread并行执行，每个Thread根据自身索引处理不同的数据。SIMT允许每个Thread独立寻址，并根据数据进入不同的条件分支或循环。与面向规则数据块、批量执行相同操作的SIMD相比，SIMT更适合表达不规则数据访问和逐Thread控制逻辑。

PyPTO Pro在AIV上提供SIMD与SIMT混合编程能力。外层Kernel组织规则的Tile计算、数据搬运和SIMT启动，SIMT函数描述逐线程逻辑，具体开发步骤见[SIMT计算](../../development/vector_computation/simt_computation.md)。

## 线程架构

### 线程层次结构

SIMT采用Grid、Thread Block和Thread三级线程层次，从顶层到底层逐级划分并行任务。Grid和Thread Block的规模，以及Thread Block和Thread的坐标，均为包含X、Y、Z三个分量的dim3三维结构。

**图1 SIMT线程层次结构**

![Grid、Thread Block、Warp与Thread的层次关系](../../../../figures/pro/simt_thread_hierarchy.png)

#### Grid（线程块网格）

Grid是SIMT线程层次结构的最顶层，由多个Thread Block组成。grid_dim = (grid_x, grid_y, grid_z)表示Grid在各维度上的Thread Block数量，各Thread Block通过自身坐标标识。

在PyPTO Pro中，Grid具有以下特点：

- 每个执行到SIMT启动点的Vector核启动一个Thread Block；pypto_pro.language.simt.grid_dim()描述外层Vector执行域，规模由Host启动Kernel时实际生效的Vector核数决定；
- 不同Thread Block彼此独立，不能依赖固定的执行顺序；
- 当前Grid仅使用X维，即grid_dim = (grid_x, 1, 1)，对应的Thread Block坐标中Y、Z均为0。

#### Thread Block（线程块）

Thread Block是Grid的组成单元，由若干Thread组成。block_dim = (block_x, block_y, block_z)表示一个Thread Block在各维度上的Thread数量，三个分量的乘积为Block内Thread总数，当前不超过2048。

Thread Block具有以下特点：

- 同一Thread Block内的Thread执行相同的SIMT入口函数，并具有相同的Thread Block尺寸；
- 块内Thread可以访问传入的共享Tile；
- 定义SIMT入口函数时，可以声明单个Thread Block允许启动的最大Thread数量，实际的Thread数由`simt_func[threads](...)`中方括号内的threads决定。

#### Thread（线程）

Thread是SIMT结构中的最小编程单元。每个Thread具有独立的局部变量和执行状态，并通过自身在Thread Block内的三维坐标处理不同数据。

当每个外层Vector核都调用同一`simt_func[threads](...)`一次时，启动的Thread总数为：

$$
threads = grid_x \times grid_y \times grid_z
\times block_x \times block_y \times block_z
$$

### Warp执行与分支

Warp是硬件在线程块内组织执行的单位，Warp Size为32，线程按块内线性编号划分到Warp；线程总数不是32的整数倍时，最后一个Warp只有部分线程有效。

同一Warp中的线程可以根据数据进入不同分支，但分支发散会降低执行效率，线程数取32的整数倍是可考虑的性能选择。

### 线程索引

每个Thread都有对应的三维坐标。开发者通过线程层级查询接口获取Grid、Thread Block和Thread的信息，从而确定当前Thread负责处理的数据。

| PyPTO Pro接口 | 说明 | 返回形式 | 约束 |
|---|---|---|---|
| pypto_pro.language.simt.grid_dim() | Grid在各维度上的Thread Block数量。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 当前仅使用X维，Y、Z维均为1；X维由外层Kernel实际使用的Vector核数决定。 |
| pypto_pro.language.simt.block_dim() | Thread Block在各维度上的Thread数量。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 三个分量的乘积不能超过入口函数的max_threads，且不能超过2048。 |
| pypto_pro.language.simt.block_idx() | 当前Thread Block在Grid中的三维坐标。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | X坐标范围由Grid的X维大小决定，当前Y、Z坐标均为0。 |
| pypto_pro.language.simt.thread_idx() | 当前Thread在Thread Block内的三维坐标。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 各维坐标范围由Thread Block对应维度的大小决定。 |
| pypto_pro.language.simt.linear_thread_idx() | 当前Thread在块内按X维优先展开的编号。 | DT_UINT32 Scalar。 | 范围为[0, 块内线程总数)，不含Block偏移。 |

**示例：定位Thread处理的数据**

假设Host使用`copy_kernel[None, 3](...)`启动3个Vector核，外层Kernel中的每个Vector核调用一次`copy_by_index[4, 2](...)`。此时，Grid包含3个Thread Block，即grid_dim为(3, 1, 1)，每个Thread Block的包含8个Thread，block_dim为(4, 2, 1)。

下例在SIMT入口函数中获取当前Thread的位置并计算它对应的全局一维索引，外层JIT Kernel负责按(4, 2, 1)的尺寸启动Thread Block：

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=8)
def copy_by_index(
    source: pl.Tensor[[1, 24], pl.DT_FP32],
    output: pl.Tensor[[1, 24], pl.DT_FP32],
):
    grid_size = pl.simt.grid_dim()
    block_size = pl.simt.block_dim()
    block_pos = pl.simt.block_idx()
    thread_pos = pl.simt.thread_idx()
    local_idx = pl.simt.linear_thread_idx()

    threads_per_block = block_size.x * block_size.y * block_size.z
    global_idx = block_pos.x * threads_per_block + local_idx
    threads_num = grid_size.x * grid_size.y * grid_size.z * threads_per_block
    output[0, global_idx] = source[0, global_idx]


@pl.jit(arch="3510")
def copy_kernel(
    source: pl.Tensor[[1, 24], pl.DT_FP32],
    output: pl.Tensor[[1, 24], pl.DT_FP32],
):
    with pl.section_vector():
        copy_by_index[4, 2](source, output)
```

以第2个Thread Block中坐标为(2, 1, 0)的Thread为例，各接口返回值及含义如下。接口返回的索引均从0开始，因此第2个Thread Block的X坐标为1。

| 接口或变量 | 当前值 | 含义 |
|---|---|---|
| grid_dim() | (3, 1, 1) | Grid在X、Y、Z维分别包含多少个Thread Block。 |
| block_dim() | (4, 2, 1) | 每个Thread Block在X、Y、Z维分别包含多少个Thread。 |
| block_idx() | (1, 0, 0) | 当前Thread位于Grid中的第2个Thread Block。 |
| thread_idx() | (2, 1, 0) | 当前Thread位于所属Thread Block的X维第3列、Y维第2行。 |
| linear_thread_idx() | 6 | 将块内坐标(2, 1, 0)按X维优先展开，计算结果为2 + 1 × 4 + 0 × 4 × 2 = 6。 |
| global_idx | 14 | 在第2个Thread Block前有8个Thread，因此全局一维索引为1 × 8 + 6 = 14。 |
| threads_num | 24 | 3个Thread Block各包含8个Thread，因此本次启动共有24个Thread。 |

如果每个Thread复制一个元素，可以使用output[0, global_idx] = source[0, global_idx]，则该Thread负责复制索引为14的元素。全部Thread共同覆盖索引0至23。

## SIMT函数

PyPTO Pro支持SIMT入口函数和SIMT辅助函数。入口函数描述Thread Block中每个Thread执行的计算逻辑；辅助函数用于复用逐Thread逻辑，在调用它的Thread中执行。

| 函数类型 | 定义方式 | 作用 | 调用方式 |
|---|---|---|---|
| SIMT入口函数 | @pypto_pro.language.vector_function(mode="simt", max_threads=N) | 定义每个Thread执行的完整计算，结果写入传入的Tile或Tensor。 | 由外层JIT Kernel通过`simt_func[threads](...)`调用。 |
| SIMT辅助函数 | @pypto_pro.language.vector_function(mode="simt") | 封装可复用的逐Thread计算，可以不返回值或返回一个Scalar。 | 由SIMT入口函数或其他辅助函数调用。 |

Host先启动外层@pypto_pro.language.jit(arch="3510") Kernel，再由外层Kernel在Vector执行域中启动SIMT入口函数；入口函数可以继续调用辅助函数。辅助函数在调用它的线程中执行，不创建新线程。具体定义和调用方式见[SIMT计算](../../development/vector_computation/simt_computation.md)。

**图2 SIMD/SIMT混合模式启动关系**

![Host启动JIT Kernel并由Vector执行域启动SIMT入口函数](../../../../figures/pro/simt_mixed_frontend_launch.png)

图中Thread 0至Thread N-1表示实际启动N个线程的情况。一般情况下，实际线程数由`simt_func[threads](...)`中方括号内的threads决定，可以小于装饰器声明的max_threads。当前不支持Host直接启动SIMT函数或在SIMT函数内嵌套调用SIMT入口函数。

## 内存层级和操作对象

SIMT函数可以操作Scalar、Tile和Tensor。三类对象对应不同的内存层级和共享范围：

| 操作对象 | 内存层级 | 作用范围 | 主要用途 |
|---|---|---|---|
| Scalar | 通常映射到寄存器 | Thread私有 | 保存参数、索引、局部变量和标量计算的中间结果。 |
| Tile | UB | Thread Block内共享 | 保存块内数据，并在Thread之间交换中间结果。 |
| Tensor | GM | Grid范围内可访问 | 保存输入、输出和跨Thread Block访问的数据，支持基于运行时索引进行不规则访存。 |
