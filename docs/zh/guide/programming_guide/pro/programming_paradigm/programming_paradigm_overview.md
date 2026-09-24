# 编程范式概述

PyPTO Pro使用Python语法编写运行在NPU上的Kernel。开发者可通过Tensor和Tile描述不同存储层级的数据，利用多核SPMD划分任务，并按计算特点选用SIMD、SIMT或组合两种并行方式。

## 并行执行模型：SIMD与SIMT

为编写高性能的Device端代码，首先需要理解底层的并行计算原理。在高性能并行编程领域，SIMD和SIMT是两种主流的并行执行模型，它们定义了指令驱动多计算单元协同工作的核心机制，是提升程序数据吞吐量、优化计算性能的关键技术，也是学习PyPTO Pro编程的核心内容。

PyPTO Pro的并行执行流程：

1. 用户在Host侧启动JIT Kernel，并通过`block_dim`配置核数。
2. 多个核以SPMD（Single Program Multiple Data，单程序多数据）方式执行同一份Kernel程序，并根据各自的核索引处理不同的数据分片。
3. 在每个逻辑执行域内，AIC主要执行Cube侧SIMD矩阵计算，AIV主要执行Vector侧SIMD或SIMT计算。
4. SIMD与SIMT可以在同一个外层Kernel中组合，并与数据搬运流水共同完成一个算子的计算。

SPMD负责组织多个逻辑AI Core之间的任务并行，SIMD和SIMT负责描述单个逻辑执行域内的计算方式，两者处于不同层级。

## SIMD（单指令多数据）

SIMD（Single Instruction Multiple Data，单指令多数据）是一种数据并行模型。核心逻辑是：一条指令在同一个时钟周期内，对多个数据元素执行完全相同的操作，实现数据的批量并行处理。

PyPTO Pro使用Tensor描述GM中的数据，使用Tile描述片上存储中的数据块。开发者以Tile或Vector Register为主要编程对象，不需要逐个描述每个数据元素的相同计算。

### 核心特征

- **单指令驱动**：所有并行计算单元同步执行同一条指令，操作完全一致；
- **数据同构**：要求参与计算的数据类型统一、长度相同，确保指令可批量处理；
- **同步执行**：所有数据的操作在同一个指令周期内完成，无独立调度逻辑，执行节奏完全统一。

### 适用场景

SIMD适合连续、规整的数据访问，以及对大量数据执行相同操作的场景，例如：

- 逐元素计算、归约和数据变换；
- 矩阵乘等高密度计算；
- 由Cube和Vector共同完成的融合计算；
- 对吞吐量和硬件资源利用率要求较高的核心计算路径。

典型的SIMD Kernel按照Tiling设计、数据搬入、数据计算和数据搬出的过程组织。详细内容参见[SIMD编程范式](SIMD/programming_paradigm.md)。

## SIMT（单指令多线程）

SIMT（Single Instruction Multiple Threads，单指令多线程）是一种线程并行模型。同一条SIMT指令由多个独立线程并行执行，每个线程通过自身索引处理不同的数据，并可独立进行地址计算和条件分支。

### 核心特征

- **多线程并行**：同一份程序由多个Thread并行执行，每个Thread处理不同的数据。
- **线程独立**：每个Thread拥有独立的索引和执行状态，可以独立寻址，并支持分支与循环。
- **并行调度**：多个Thread由硬件并行调度，不同Thread可以采用不同的控制路径。

SIMT编程先定义SIMT入口函数，再由外层JIT Kernel在Vector执行域中启动Thread Block。

### 适用场景

SIMT适合需要逐线程控制或难以使用规整Tile计算表达的场景，例如：

- 离散索引和不规则数据访问；
- 分支较多或不同数据需要执行不同处理逻辑的计算；
- 直方图、计数器等需要原子更新的计算；
- 稀疏数据处理和动态数据依赖较强的局部计算。

典型的SIMT计算先定义SIMT函数，通过线程索引划分数据，再使用标量计算、同步或原子接口完成逐Thread处理。详细内容参见[SIMT编程范式](SIMT/programming_paradigm.md)。

## AI Core硬件基础

AI处理器包含多个AI Core，多个AI Core可以并行处理不同的数据分片。AI Core内部包含Scalar、Vector、Cube、片上存储和数据搬运等单元：

| 硬件组成 | 主要职责 | PyPTO Pro中的对应表达 |
|---|---|---|
| Scalar单元 | Kernel控制流、地址计算和指令调度 | Python控制流、标量表达式和数据索引计算 |
| Vector单元 | 矢量计算和SIMT线程计算 | Vector执行域中的Tile API、VF函数和SIMT函数 |
| Cube单元 | 矩阵乘加等矩阵计算 | Cube执行域中的矩阵计算接口 |
| 片上存储 | 保存计算输入、输出和中间数据 | 不同内存空间中的Tile |
| 数据搬运单元 | 在GM与片上存储、不同片上存储之间搬运数据 | 数据搬入、片上搬运和结果写回操作 |

计算与搬运任务在不同流水上执行，存在数据依赖时需要通过同步机制约束执行顺序。

硬件单元、存储层级及其接口映射参见[SIMD抽象硬件架构](SIMD/abstract_hardware_architecture.md)和[SIMT抽象硬件架构](SIMT/abstract_hardware_architecture.md)。

## 开发流程与学习路径

PyPTO Pro算子的典型开发与运行流程如下：

1. 在Host侧使用PyTorch准备输入、输出和工作空间Tensor。
2. 使用JIT装饰器定义Kernel，并声明Tensor、Scalar或TilingData等参数。
3. 根据数据规模设计多核切分和Tiling，规划Tile及片上存储。
4. 根据数据访问模式和控制流特点选择SIMD、SIMT，或在同一个Kernel中组合两种计算方式。
5. 在Host侧启动Kernel，并按需配置Stream和`block_dim`；首次调用触发JIT编译。
6. Kernel异步下发后，在Host读取结果前等待Device任务完成。

建议继续阅读以下内容：

- [PyPTO Pro快速入门](../../../quick_start/pro/index.md)：通过完整算子示例了解Kernel定义、编译和运行流程。
- [SIMD编程](SIMD/index.md)：了解SPMD多核并行、Tile编程、SIMD矢量计算和矩阵计算。
- [SIMT编程](SIMT/index.md)：了解线程架构、内存层级、SIMT函数、同步和原子操作。
- [Kernel核函数](../development/kernel_function.md)：了解Kernel参数、实际核数和启动方式。
- [编译与执行](../development/compilation_and_execution/index.md)：了解JIT编译和离线二进制编译。
