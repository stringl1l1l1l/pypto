# pypto_pro.language.MemorySpace

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

内存空间枚举，用于描述PyPTO数据对象对应的硬件存储区域。创建片上Tile时，该枚举作为[pypto_pro.language.TileType](TileType.md)的target_memory属性，用于指定Tile的目标内存空间。

不同的枚举值对应AI处理器的全局存储或AI Core内的片上存储区域。对于片上Tile，内存空间决定其可参与的计算类型和支持的数据搬运路径。

## 原型定义

```python
class MemorySpace(enum.IntEnum):
    DDR = ...
    Vec = ...
    Mat = ...
    Left = ...
    Right = ...
    Scaling = ...
    Acc = ...
    Bias = ...
    ScaleLeft = ...
    ScaleRight = ...
```

## 参数说明

| 参数值 | 存储单元 | 用法与约束 |
|---|---|---|
| pypto_pro.language.MemorySpace.DDR | GM | 表示Kernel输入、输出Tensor所在的存储区域。该枚举值用于表示Tensor存储位置，不用作make_tile/make_tile_group的片上Tile分配目标。 |
| pypto_pro.language.MemorySpace.Vec | UB | Vector Core的工作缓冲区，用于逐元素、归约、排序、纯Vector量化/反量化等SIMD计算。Tile起始地址须按32字节对齐。 |
| pypto_pro.language.MemorySpace.Mat | L1 Buffer | 用于存放Cube计算的中间数据。数据从GM加载后可搬入L0A Buffer、L0B Buffer、BiasTable Buffer、Fixpipe Buffer、L0A_MX Buffer和L0B_MX Buffer。Tile起始地址须按32字节对齐。 |
| pypto_pro.language.MemorySpace.Left | L0A Buffer | 用于存放矩阵乘的左操作数，与L0B Buffer中的右操作数共同参与Cube计算。Tile起始地址须按512字节对齐。 |
| pypto_pro.language.MemorySpace.Right | L0B Buffer | 用于存放矩阵乘的右操作数，与L0A Buffer中的左操作数共同参与Cube计算。Tile起始地址须按512字节对齐。 |
| pypto_pro.language.MemorySpace.Scaling | Fixpipe Buffer | 仅用于FIX数据通路的per-channel随路量化/反量化参数。该空间不是quant/dequant接口的scale存储区。数据需经L1 Buffer搬入：源Tile须为1行，目的Tile数据类型须为DT_INT64或DT_UINT64，目的地址和数据量均须按128字节对齐，单次搬运数据量不得超过4096字节。 |
| pypto_pro.language.MemorySpace.Acc | L0C Buffer | 用于存放matmul、matmul_acc、matmul_mx等Cube计算的结果或中间累加值。Tile起始地址须按64字节对齐。 |
| pypto_pro.language.MemorySpace.Bias | BiasTable Buffer | 用于存放矩阵计算的偏置数据。偏置数据不能从GM直接加载，需经L1 Buffer中转。经L1 Buffer搬入时，源Tile须为1行，目的地址和数据量均须按64字节对齐，单次搬运数据量不得超过4096字节。 |
| pypto_pro.language.MemorySpace.ScaleLeft | L0A_MX Buffer | 仅用于存放MX矩阵乘中左量化系数矩阵。Tile起始地址须按32字节对齐；其地址必须等于配套左矩阵在L0A Buffer中的地址右移4位。 |
| pypto_pro.language.MemorySpace.ScaleRight | L0B_MX Buffer | 仅用于存放MX矩阵乘中右量化系数矩阵。Tile起始地址须按32字节对齐；其地址必须等于配套右矩阵在L0B Buffer中的地址右移4位。 |
