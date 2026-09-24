# Tiling概述

Kernel在Device侧由多个AI Core并行执行，而每个AI Core的片上Buffer容量有限。为了用满多核算力，并让单个核处理的数据能放进片上Buffer，需要把算子的输入数据划分为基本任务块，将一组任务块分配给各个核，再为核内处理配置Buffer。这个对数据进行切分、分块计算的过程称为Tiling。

Tiling是算子开发的第一步：切分方式决定了各个核的负载是否均衡、片上Buffer能否放下、数据搬运是否连续，直接影响算子的性能。

## 为什么需要Tiling

两个硬件约束决定了Tiling的必要性。

**Device上有多个AI Core。** 算子的数据如果只交给一个核处理，其余核处于空闲，多核算力无法发挥。把数据划分成基本任务块，并将一组任务块分配给每个核并行处理，才能把算力用起来。

**单个核的片上Buffer容量有限。** 计算只能在片上Buffer中进行，而算子的输入输出通常远大于片上容量，无法一次完整装入。基本任务块的大小需要适配片上容量；核内可为已分配的任务划分双缓冲或多个Buffer槽位，轮转执行“搬入—计算—搬出”，让不同任务的搬运与计算尽量并行。

因此Tiling包含两个层次：

| 层次 | 含义 | 详细内容 |
| --- | --- | --- |
| 核间切分 | 把算子的数据划分为基本任务块，将一组任务块分配给每个AI Core并行处理。 | [多核Tiling切分](multi_core_tiling.md) |
| 核内切分 | 基于已分配的任务划分核内Buffer，例如使用双缓冲或多个Buffer槽位轮转，让搬运与计算并行。 | [Tile创建和操作](../tile_creation_and_operations.md)、[Tile计算](../vector_computation/tile_computation.md) |

**图1 Tiling的两个层次**

![Tiling的两个层次](../../../../figures/pro/pro_tiling_two_levels.png "Tiling的两个层次")

图中AI Core 0负责完整的T0、T4、T8三个基本任务块；核内示意的是这些任务对Buffer槽位的轮转使用，并非把T0再次切成更小的Tile。

## AI Core与执行域

Device上有多个AI Core，每个AI Core内含Cube核（AIC）和Vector核（AIV）。在AIC与AIV为1:2的芯片上，一个AI Core内有1个Cube核和2个Vector核：Cube核承担矩阵计算，Vector核承担矢量计算，两者各有自己的片上Buffer和指令流水。

Kernel通过[执行域](../kernel_function.md#定义执行域)声明代码运行在哪类核上：`pypto_pro.language.section_cube()`中的代码由Cube核执行，`pypto_pro.language.section_vector()`中的代码由Vector核执行。Kernel包含哪些执行域，决定了它使用哪类核，也决定了核间切分时可用的核数。

**图2 AI Core与执行域**

![AI Core与执行域](../../../../figures/pro/pro_tiling_ai_core.png "AI Core与执行域")

片上Buffer的层级、容量和数据通路参见[SIMD抽象硬件架构](../../programming_paradigm/SIMD/abstract_hardware_architecture.md)。

## Tiling参数的产生与传递

切分需要的参数（每个核处理多少任务、每个Tile多大、循环多少次等）通常依赖运行时的shape，无法在编写Kernel时写死，需要在Host侧计算后传给Kernel。这些参数由TilingKey和TilingData共同承担：TilingKey在编译时决定走哪套实现，TilingData在运行时提供该实现所需的数值。

两者的分工、声明方式和传递规则参见[Tiling参数定义与传递](tiling_parameter_definition.md)。

## 设计Tiling时的考虑因素

- **负载均衡**：各个核分到的任务数和计算量应尽量接近，避免个别核成为长尾拖慢整体。
- **片上容量**：单次搬入的Tile必须能放进对应的片上Buffer，并为双缓冲等优化预留空间。
- **接口约束**：Tile的形状需要满足数据搬运和计算接口在对齐、layout和分形上的要求。
- **数据复用**：相邻任务如果复用同一块数据，切分顺序应尽量让它们落在同一个核上。
- **尾块处理**：数据量通常不能被Tile大小整除，任务数需要向上取整，最后一块用有效形状表达实际范围。

## 相关内容

- [多核Tiling切分](multi_core_tiling.md)：核间切分的具体写法，包括任务编号与连续分核、MIX Kernel的切分和负载均衡。
- [Tiling参数定义与传递](tiling_parameter_definition.md)：TilingData和TilingKey的声明、传递与配合使用。
- [Tile创建和操作](../tile_creation_and_operations.md)：Tile的规格描述、片上地址绑定和有效形状设置。
- [SIMD抽象硬件架构](../../programming_paradigm/SIMD/abstract_hardware_architecture.md)：AIC与AIV的计算单元、片上存储层级和数据通路。
