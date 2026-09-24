# 抽象硬件架构

PyPTO Pro的SIMD计算使用AIV的Vector资源和AIC的Cube资源。理解这两类计算资源及其数据通路，有助于选择Tile计算、Reg计算或Cube计算，并正确组织数据搬运与同步。SIMT的线程执行资源参见[SIMT抽象硬件架构](../SIMT/abstract_hardware_architecture.md)。

## 概览

AI Core的相关硬件可从计算单元、存储单元和搬运单元三个方面理解。下图展示AIC、AIV及其指令流和主要数据流。

**图1 AI Core硬件架构（Ascend 950PR&950DT系列产品）**

![AIC、AIV、存储、计算和搬运单元的关系](../../../../figures/pro/hardware_architecture_950.png)

图中黑色实线表示主要数据流，橙色虚线表示指令流。图中的SIMT资源用于说明其在AIV中的位置，不属于本章讨论范围。

## 计算单元

| 单元 | 主要作用 | PyPTO Pro计算方式 |
|:---|:---|:---|
| Scalar | 处理地址计算、循环控制等标量逻辑，向计算和搬运流水发射指令 | Kernel中的控制逻辑 |
| AIV上的Vector单元 | 执行矢量运算 | [Tile计算](../../development/vector_computation/tile_computation.md)、[Reg计算](../../development/vector_computation/reg_computation.md) |
| AIC上的Cube单元 | 执行矩阵运算 | [Cube计算](../../development/cube_computation.md) |

Tile计算以UB中的Tile为计算数据载体；Reg计算将UB中的数据加载到Vector Register File，中间结果可以保留在寄存器中；Cube计算从L0A Buffer和L0B Buffer读取矩阵操作数，将累加结果写入L0C Buffer。各方式的接口和使用约束见对应的编程章节。

## 存储单元与搬运单元

GM位于AI Core之外，用于保存Kernel输入、输出及Workspace。AIV和AIC各有片上存储，计算前需要将数据搬入相应的存储层级。

| 执行侧 | 主要片上存储 | 典型数据路径 |
|:---|:---|:---|
| AIV Tile计算 | UB | GM → UB → Vector单元 → UB → GM |
| AIV Reg计算 | UB、Vector Register File | GM → UB → Vector Register File → UB → GM |
| AIC Cube计算 | L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer | GM → L1 Buffer → L0A Buffer/L0B Buffer → Cube单元 → L0C Buffer → GM |

表中的路径表示一次计算所涉及的主要存储层级，不表示所有搬运和计算必须串行完成。Reg计算的寄存器数据路径及硬件单元见[Reg计算](../../development/vector_computation/reg_computation.md#硬件组成)；AIC各Buffer的作用、布局和搬运路径见[Cube计算](../../development/cube_computation.md)。

MTE等搬运单元负责GM与片上Buffer以及不同片上Buffer之间的数据流转。AIV的GM与UB搬运主要使用MTE2、MTE3；Reg计算还使用DMA单元在UB与Vector Register File之间搬运数据。AIC的矩阵数据通常经MTE2进入L1 Buffer，再经MTE1进入L0A Buffer、L0B Buffer；FIX负责结果搬出。具体接口及支持的搬运路径以[Tile创建和操作](../../development/tile_creation_and_operations.md)、[Reg计算](../../development/vector_computation/reg_computation.md)和[Cube计算](../../development/cube_computation.md)为准。

## 指令流、数据流与同步

图1中的指令流与数据流含义不同：Scalar将搬运和计算任务发射到相应流水；搬运单元和计算单元按各自的数据路径访问存储资源。不同流水可以异步执行，因此源码中的先后顺序不等同于硬件上的完成顺序。

当一条流水生产的数据要由另一条流水消费时，需要建立同步依赖。例如，Vector计算读取UB数据之前，相关搬入必须完成；将计算结果搬回GM之前，相关计算也必须完成。同步信号只约束执行顺序，不承担数据搬运。

PyPTO Pro可通过TileGroup与自动mutex管理依赖，也可使用同步接口显式指定依赖。具体使用方式参见[Tile创建和操作](../../development/tile_creation_and_operations.md)和[Tile计算](../../development/vector_computation/tile_computation.md)。

多个AIC或AIV可以并行处理不同数据分片。核数配置、核索引以及混合Kernel中AIC与AIV的映射关系，参见[SIMD编程范式](programming_paradigm.md#多核spmd与核内simd)和[多核Tiling切分](../../development/tiling/multi_core_tiling.md)。
