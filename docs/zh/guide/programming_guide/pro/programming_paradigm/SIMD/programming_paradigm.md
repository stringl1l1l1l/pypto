# 编程范式

SIMD（Single Instruction Multiple Data，单指令多数据）以数据块为主要编程对象，一条指令同时对多个同构数据元素执行相同操作。PyPTO Pro使用Tensor、Tile和RegTensor表示不同存储层级中的数据，并将多核数据切分和核内批量计算组织在同一个Kernel中。

PyPTO Pro的SIMD编程包含三种主要计算方式：基于UB Tile的Tile计算、基于Vector Register的Reg计算，以及基于L1 Buffer和L0 Buffer Tile的Cube计算。三者都以批量数据为操作对象，适合访存和计算较规整、并行度较高的算子。

## 多核SPMD与核内SIMD

PyPTO Pro SIMD Kernel采用“外层多核SPMD + 内层单核SIMD”的两级并行方式：

- **多核SPMD（Single Program Multiple Data）**：多个AI Core执行同一份Kernel代码，各核通过核索引处理不同的数据分片。
- **核内SIMD**：每个AI Core使用Vector或Cube指令批量处理当前数据分片中的多个元素。

Host侧通过`block_dim`配置核数；Kernel内使用`pypto_pro.language.get_block_num()`取得实际生效核数，使用`pypto_pro.language.get_block_idx()`取得当前执行域的核索引。`block_dim`的默认值、调用形式和不同Kernel类型下的含义参见[Kernel核函数](../../development/kernel_function.md#blockdim的含义与设置)。

```python
import pypto_pro.language as pl


num_cores = pl.get_block_num()
core_id = pl.get_block_idx()

# 第core_id个核以num_cores为步长处理数据块。
for tile_idx in pl.range(core_id, tile_num, num_cores):
    ...
```

常用的跨步切分让各核处理的数据块数量最多相差1，并避免由单个核串行遍历全部数据。完整的Block数计算和多核切分方法请参考[多核Tiling切分](../../development/tiling/multi_core_tiling.md)。

### 纯Vector、纯Cube与混合Kernel

SIMD Kernel可以只使用一种执行域，也可以组合Vector和Cube执行域：

| Kernel类型 | 执行资源 | `get_block_idx()`的含义 |
|:---|:---|:---|
| 纯Vector Kernel | AIV | 当前Vector核的全局索引 |
| 纯Cube Kernel | AIC | 当前Cube核的全局索引 |
| Cube/Vector混合Kernel | AIC和AIV | 在各自执行域中返回相应的全局核索引 |

混合Kernel采用AIC:AIV为1:2的映射。每个AI Core执行组包含一个AIC和两个AIV；当`block_num = pypto_pro.language.get_block_num()`时，Cube执行域实际使用`block_num`个AIC，Vector执行域实际使用`2 * block_num`个AIV。

**图1 混合Kernel的执行组映射**

![AIC与AIV为1比2时的执行组和执行域映射](../../../../figures/pro/pro_multicore_spmd_mapping.png)

Vector执行域中的`get_block_idx()`已经是展平后的全局AIV索引。需要区分同一个执行组内的两个AIV时，可以使用`pypto_pro.language.get_subblock_idx()`。

## SIMD数据对象

PyPTO Pro使用不同的数据对象表示SIMD数据在存储层级中的位置：

| 数据对象 | 数据位置 | 作用 |
|:---|:---|:---|
| Tensor | GM | 描述Kernel输入、输出或Workspace中的多维数据视图 |
| Tile | UB、L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer等片上存储 | 表示当前分块的数据，是Tile计算和Cube计算的操作数 |
| TileGroup | 与Tile相同 | 管理一组轮转Tile，用于单缓冲、双缓冲和N缓冲 |
| RegTensor、MaskReg | Vector Register File | 保存Reg计算的输入、中间结果和输出 |

Tensor表示全局数据，Tile表示当前AI Core处理的局部数据块。开发者通过Tiling将大Tensor划分成多个Tile，再由不同核和不同循环迭代处理这些Tile。Tensor和Tile的创建方式分别参见[Tensor创建和操作](../../development/tensor_creation_and_operations.md)和[Tile创建和操作](../../development/tile_creation_and_operations.md)。

## SIMD计算方式

### Tile计算

Tile计算以UB中的二维Tile作为计算对象，在`pypto_pro.language.section_vector()`执行域中完成批量运算。Tile API适合逐元素、归约、数据类型转换和数据重排等通用矢量场景。

Tile分配、数据搬运、计算接口、缓冲区轮转和尾块处理请参考[Tile计算](../../development/vector_computation/tile_computation.md)；完整可执行示例请参考[Softmax算子快速入门（SIMD）](../../../../quick_start/pro/softmax_simd.md)。

### Reg计算

Reg计算也称Regbase矢量计算，通过`@pypto_pro.language.vector_function`定义VF函数，并在函数内使用`vf.*`接口操作Vector Register File中的RegTensor和MaskReg。中间结果可以保留在寄存器中，适合计算链较长、需要减少UB往返访问的高性能场景。

VF函数不能独立启动，需要由外层JIT Kernel在Vector执行域中调用。寄存器数据类型、VF函数、加载存储和计算接口的完整规则请参考[Reg计算](../../development/vector_computation/reg_computation.md)。

### Cube计算

Cube计算使用L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer中的矩阵Tile，通过一条矩阵指令并行完成一个矩阵分块的乘加运算。Kernel使用`pypto_pro.language.section_cube()`标识Cube执行域。

开发者根据矩阵分块选择Tile shape并组织矩阵计算。矩阵分形、片上地址、数据搬运、计算接口和完整示例请参考[Cube计算](../../development/cube_computation.md)。

## SIMD Kernel开发流程

使用PyPTO Pro开发SIMD算子通常包含以下步骤：

1. **确定计算方式**：根据数据访问和计算特点选择Tile计算、Reg计算或Cube计算。
2. **设计多核与Tile切分**：确定`block_dim`、各核的数据范围、Tile shape和尾块处理方式。
3. **声明Tensor参数**：在Kernel签名中描述GM输入、输出和Workspace的数据类型、shape及layout。
4. **规划片上数据**：根据计算方式选择Tile所在的MemorySpace和数据排布。
5. **编写执行域**：在`section_vector()`或`section_cube()`中表达相应的SIMD计算。
6. **编译和启动**：使用`@pypto_pro.language.jit`编译Kernel，在Host侧通过`kernel[stream, block_dim](...)`启动。

SIMD适合对连续或规则分块数据执行相同操作。若算法更适合逐线程索引、不规则访存、复杂分支或原子更新，应考虑[SIMT编程范式](../SIMT/programming_paradigm.md)。
