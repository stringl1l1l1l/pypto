# Cube计算

本文介绍如何在PyPTO Pro中使用Tile API编写基于L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer等片上存储的**矩阵计算代码**。

## 矩阵编程的基本步骤

一个常见的Cube计算步骤如下：

1. 分别创建L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer。
2. 将左、右矩阵从GM搬入L1 Buffer。
3. 将L1 Buffer上的左矩阵数据搬入L0A Buffer，右矩阵数据搬入L0B Buffer。
4. 执行矩阵计算，并将结果写入L0C Buffer。
5. 将L0C Buffer中的结果写回GM。
6. 正确配置同步指令，确保搬运与计算步骤不发生数据竞争。

以上步骤需要写在[pypto_pro.language.section_cube()](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md)标记的Cube执行域中，Tile创建可以放在执行域外部。完整的Kernel代码参考本文末尾的[完整示例](#完整示例)。

对应的数据流和硬件流水如下：

**图1 Cube计算的数据流和硬件流水**

![Cube计算的数据流和硬件流水](../../../figures/pro/cube_matrix_computation_data_flow.png "Cube计算的数据流和硬件流水")

## 矩阵计算内存管理

### Cube侧Tile创建

输入数据从GM搬入L1 Buffer，再从L1 Buffer搬入L0A Buffer和L0B Buffer，经Cube计算后，将结果写入L0C Buffer。

PyPTO Pro通过[TileType](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)描述Tile的shape、dtype和target_memory。TileType本身不创建Tile，需要将其传给[pypto_pro.language.make_tile](../../../../api/pro_api/SIMD-API/resource_management/make_tile.md)或[pypto_pro.language.make_tile_group](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)，将Tile绑定到指定Buffer的地址。

本节代码仅展示Tile创建和同步方式，省略了完整Kernel的计算、调用及结果验证代码；整体代码结构及调用方式请参考本文末尾的[完整示例](#完整示例)。

各内存空间的典型角色如下：

| pypto_pro.language.MemorySpace | 物理缓冲区 | 典型角色 |
|:---|:---|:---|
| Mat | L1 Buffer | GM与L0A Buffer/L0B Buffer之间的矩阵暂存 |
| Left | L0A Buffer | matmul左操作数 |
| Right | L0B Buffer | matmul右操作数 |
| Acc | L0C Buffer | matmul累加结果（通常为FP32/INT32） |

#### 使用make_tile创建单个Tile

pypto_pro.language.make_tile将Tile绑定到一段固定的片上缓冲区地址。addr必须指定；地址范围的字节数根据TileType的shape和dtype自动推导。使用make_tile时，跨Pipe依赖通过显式的[pypto_pro.language.system.sync_src](../../../../api/pro_api/SIMD-API/synchronization/sync_src.md)/[pypto_pro.language.system.sync_dst](../../../../api/pro_api/SIMD-API/synchronization/sync_dst.md)对进行同步。

```python
import pypto_pro.language as pl


TILE_M = 128
TILE_K = 128
TILE_N = 128

a_l1_type = pl.TileType(
    shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
b_l1_type = pl.TileType(
    shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
a_left_type = pl.TileType(
    shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
b_right_type = pl.TileType(
    shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
acc_type = pl.TileType(
    shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)

a_l1 = pl.make_tile(a_l1_type, addr=0x00000)
b_l1 = pl.make_tile(b_l1_type, addr=0x08000)
a_left = pl.make_tile(a_left_type, addr=0x0000)
b_right = pl.make_tile(b_right_type, addr=0x0000)
acc = pl.make_tile(acc_type, addr=0x0000)
```

#### 使用make_tile_group创建轮转Tile

pypto_pro.language.make_tile_group创建一组绑定不同片上地址的轮转Tile。组内Tile数量由depth指定；未指定depth且mutex_ids非空时，由mutex_ids的长度确定。可通过next()、current()和previous()选择Tile。配合`@pypto_pro.language.jit(auto_mutex=True)`时，框架根据每个Tile的mutex_id自动插入跨Pipe同步。单缓冲也可以使用长度为1的mutex_ids，从而复用自动同步机制。

```python
import pypto_pro.language as pl


TILE_M = 128
TILE_K = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def matmul_kernel(a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
                  b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
                  out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    # L1 Buffer使用双缓冲。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])

    # L0A Buffer、L0B Buffer和L0C Buffer使用单缓冲，并由auto_mutex管理同步。
    a_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])
```

两种分配方式的区别如下：

| 方面 | make_tile | make_tile_group |
|:---|:---|:---|
| 缓冲区组织 | 单块固定缓冲区，必须指定addr，size可选 | 一组轮转缓冲区，使用addrs和mutex_ids |
| 缓冲区选择 | 直接使用Tile变量 | 通过next()、current()、previous()选择 |
| 跨Pipe同步 | 手动插入sync_src/sync_dst | 带mutex元数据且配合`auto_mutex=True`时自动插入 |
| 适用场景 | 需要精确控制同步时序 | 单缓冲、双缓冲及N缓冲等常规场景 |

常规单缓冲、双缓冲及N缓冲场景使用make_tile_group并启用auto_mutex=True；需要精确控制同步事件及插入位置的场景使用make_tile和显式同步。

### 矩阵计算分形介绍

#### Ascend Cube分形布局

AI处理器Cube计算单元以分形块（Fractal）作为基本计算和搬运单位。传统ND线性布局按行连续存放矩阵，读取二维计算块时需要从多段地址逐行收集数据。分形布局通过搬运流水重排数据，使计算块在物理内存中连续存放，从而减少寻址开销并提高数据吞吐率。

输入分形的一边固定为16，另一边为`32B / sizeof(dtype)`。常用数据类型的分形大小为：

- FP32：A矩阵为16×8，B矩阵为8×16。
- FP16/BF16：A、B矩阵均为16×16。
- FP8（E4M3FN/E5M2）：A矩阵为16×32，B矩阵为32×16，累加结果为FP32。
- HF8：A矩阵为16×32，B矩阵为32×16，累加结果为FP32。
- INT8：A矩阵为16×32，B矩阵为32×16，累加结果为INT32。

L0C Buffer中的结果分形固定为16×16。以FP32/INT32累加结果为例，一个结果分形占用`16 × 16 × 4B = 1024B`。

下图以FP16类型的40×56矩阵为例，展示compact=0时的标准分形布局：GM中的ND Tensor搬入L1 Buffer中的Tile，并转换为NZ布局。有效区为40×56，按16×16分形对齐后的寻址边界为48×64；分形之间按列优先排列，分形内部按行优先排列。图中白色区域为有效数据，灰色区域为无效区域，其值未必为0。[pypto_pro.language.set_validshape](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md#参数说明)设置Tile的有效区域，[pad](../../../../api/pro_api/SIMD-API/basic_data_structures/TilePad.md)和[pypto_pro.language.fillpad](../../../../api/pro_api/SIMD-API/memory_data_movement/load.md#搬运尾块)决定是否以及如何填充无效区域；compact=1会按valid shape紧凑解释片上布局，不使用图2所示的完整标准分形边界。

**图2 pypto_pro.language.load完成ND到NZ的分形转换**

![PyPTO Pro中pypto_pro.language.load完成ND到NZ的分形转换](../../../figures/pro/cube_matrix_nd_to_nz.png "PyPTO Pro中pypto_pro.language.load完成ND到NZ的分形转换")

#### 分形格式的命名

矩阵分形格式采用“大Y小x”命名法：

- 大Y（Z/N）表示多个分形之间的排列顺序：Z为行主序，N为列主序。
- 小x（z/n）表示一个分形内部的元素排列顺序：z行主序，n为列主序。

以二维矩阵为例，几种常用格式的含义如下：

- **ND**：通用线性布局，通常用于GM中的输入和输出Tensor。
- **NZ**：分形之间按列主序排列，分形内部按行主序排列。对shape为[M, N]的矩阵，补齐并拆分为[M1, M0, N1, N0]后，物理排列顺序为[N1, M1, M0, N0]。
- **ZN**：分形之间按行主序排列，分形内部按列主序排列。对shape为[K, N]的矩阵，补齐并拆分为[K1, K0, N1, N0]后，物理排列顺序为[K1, N1, N0, K0]。

ND、NZ和ZN等数据排布格式的详细说明请参见[数据排布格式](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md)。

对于矩阵乘法`C = A × B`，左矩阵A使用NZ，右矩阵B使用ZN，结果矩阵C使用NZ。左矩阵按行取数、右矩阵按列取数时，相应元素均能从连续地址读取。

默认数据路径如下：

- GM中的ND数据搬入L1 Buffer时转换为NZ。
- A矩阵从L1 Buffer搬入L0A Buffer后保持NZ。
- B矩阵从L1 Buffer搬入L0B Buffer时转换为ZN。
- matmul的结果在L0C Buffer中按NZ存放。

layout描述Tile的物理排布，TileType.shape保持逻辑轴语义；例如B矩阵在L0B Buffer中仍使用[K, N]描述shape，物理布局为ZN。转置搬入等特殊场景需要显式指定layout。

下图以FP16输入、FP32累加为例，展示L0A Buffer、L0B Buffer和L0C Buffer中的Tile与PyPTO Pro接口的对应关系。

**图3 矩阵乘法的NZ × ZN = NZ分形组合（FP16输入）**

![PyPTO Pro矩阵乘法的NZ × ZN = NZ分形组合](../../../figures/pro/cube_matrix_fractal_formats_950.png "PyPTO Pro矩阵乘法的NZ × ZN = NZ分形组合")

### Cube侧同步

Cube计算的四个步骤分别对应MTE2、MTE1、M、FIX四条流水线。各流水线异步执行，当一条流水线生产的数据被另一条流水线消费时，需要插入同步以保证数据依赖。

| 流水线 | 含义 | 典型操作 |
|:---|:---|:---|
| MTE2 | GM→L1 Buffer搬运 | pypto_pro.language.load/pypto_pro.language.load_tile |
| MTE1 | L1 Buffer→L0A Buffer/L0B Buffer搬运 | pypto_pro.language.move |
| M | 矩阵计算 | pypto_pro.language.matmul/pypto_pro.language.matmul_acc |
| FIX | L0C Buffer→GM搬运 | pypto_pro.language.store/pypto_pro.language.store_tile |

使用make_tile_group并通过`@pypto_pro.language.jit(auto_mutex=True)`启用自动同步时，框架根据Tile的使用关系和mutex_id插入mutex_lock/mutex_unlock。

使用make_tile时，框架不会自动插入跨Pipe同步，需要在生产操作之后、消费操作之前插入配对的pypto_pro.language.system.sync_src和pypto_pro.language.system.sync_dst。下面展示一次完整矩阵计算中的前向数据依赖：

```python
import pypto_pro.language as pl


with pl.section_cube():
    pl.load(a_l1, a, [0, 0])
    pl.load(b_l1, b, [0, 0])
    pl.system.sync_src(
        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.move(a_left, a_l1)
    pl.move(b_right, b_l1)
    pl.system.sync_src(
        set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

    pl.matmul(acc, a_left, b_right)
    pl.system.sync_src(
        set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
    pl.store(out, acc, [0, 0])
```

sync_src由生产流水线SET flag，sync_dst由消费流水线WAIT flag；两者的set_pipe、wait_pipe和event_id必须一致。静态event_id取值范围为[0, 7]；动态整数Scalar的运行时数值也必须在该范围内。同一ID只能在上一次同步已经消费后复用。循环复用Tile时还需处理消费完成后才能覆盖缓冲区的反向依赖；常规流水化场景推荐使用make_tile_group和自动同步。

> [!NOTE]说明
> 当matmul/matmul_acc使用了phase参数时，M流水与FIX流水之间的同步由硬件unit_flag完成，框架不会自动插入该段同步。

## 矩阵数据搬入

### GM → L1 Buffer搬运

通过pypto_pro.language.load将矩阵从GM搬入L1 Buffer。load在搬运过程中自动完成ND到NZ的格式转换，无需手动配置分形参数。

```python
import pypto_pro.language as pl


with pl.section_cube():
    cur_a = a_l1.current()
    cur_b = b_l1.current()
    pl.load(cur_a, a, [i, 0])    # A矩阵搬入L1 Buffer，自动ND→NZ
    pl.load(cur_b, b, [0, j])    # B矩阵搬入L1 Buffer，自动ND→NZ
```

load的坐标参数[row, col]为GM Tensor上的元素偏移，表示从该位置开始搬运一个Tile大小的数据。

### L1 Buffer → L0A Buffer/L0B Buffer搬运

通过pypto_pro.language.move将L1 Buffer中的数据搬入L0A Buffer或L0B Buffer，搬运过程中自动完成NZ到ZN（L0B Buffer）的格式转换。

```python
import pypto_pro.language as pl


cur_a_left = a_left.current()
cur_b_right = b_right.current()
pl.move(cur_a_left, cur_a)    # L1 Buffer → L0A Buffer，NZ→NZ
pl.move(cur_b_right, cur_b)   # L1 Buffer → L0B Buffer，NZ→ZN
```

### 转置搬入

当GM输入矩阵的轴序与L1 Buffer中的Tile的轴序相反时（如Tensor为[K, M]而Tile为[M, K]），需要在搬入时进行转置。通过load的order参数控制：order=[1, 0]表示转置搬入，此时L1 Buffer的layout需设为ZN。

以`C[M, N] = A[M, K] @ B[K, N]`为例：

| 操作数 | Tensor shape | 是否转置 | load的order | L1 Buffer Tile layout |
|:---|:---|:---|:---|:---|
| 左矩阵A | [M, K] | 否 | [0, 1]（默认） | NZ（默认） |
| 左矩阵A | [K, M] | 是 | [1, 0] | ZN |
| 右矩阵B | [K, N] | 否 | [0, 1]（默认） | NZ（默认） |
| 右矩阵B | [N, K] | 是 | [1, 0] | ZN |

左矩阵转置搬入示例：

```python
import pypto_pro.language as pl


M = 64
K = 128
N = 64


@pl.jit(auto_mutex=True)
def kernel_left_transpose(
    a: pl.Tensor[[K, M], pl.DT_FP16],               # [K, M]，需转置
    b: pl.Tensor[[K, N], pl.DT_FP16],               # [K, N]，不转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0], order=[1, 0])    # 转置搬入
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0])                   # 不转置
```

## 矩阵数据搬出

通过pypto_pro.language.store将L0C Buffer中的计算结果搬出到GM，支持在搬运过程中自动进行格式转换（如NZ → ND）。

```python
import pypto_pro.language as pl


pl.store(out, acc, [i, j])    # L0C Buffer → GM，自动NZ→ND
```

如果输出Tensor标注为NZ，store会将L0C Buffer的计算结果按NZ分形直接写入GM，无需额外格式转换：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def kernel(
    nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],
):
    # acc为计算得到的L0C Buffer Tile，其他过程省略
    ...
    pl.store(nz_out, acc, [0, 0])    # 按NZ分形写入GM
```

## 矩阵计算

[pypto_pro.language.matmul](../../../../api/pro_api/SIMD-API/cube_computation/matmul.md)是PyPTO Pro封装NPU硬件计算能力的矩阵乘法核心接口，实现`dst_tile = lhs_tile × rhs_tile`，数据通路为L0A Buffer × L0B Buffer → L0C Buffer。

**表：矩阵乘计算A、B、C矩阵说明**

| 矩阵 | 存储位置 | 维度 | 数据格式 | 数据类型 |
|:---|:---|:---|:---|:---|
| A | L0A Buffer | M × K | NZ | FP16、BF16、FP32、INT8、HF8、FP8E4M3FN或FP8E5M2 |
| B | L0B Buffer | K × N | ZN | 与A组成接口支持的数据类型组合 |
| C | L0C Buffer | M × N | NZ | FP32或INT32，由输入数据类型组合决定 |

```python
import pypto_pro.language as pl


pl.matmul(acc_tile, a_left, b_right)    # C = A × B
```

### MXFP8/MXFP4矩阵乘

MX矩阵乘使用pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc，除L0A Buffer/L0B Buffer的Tile外，还需要分别位于L0A_MX Buffer和L0B_MX Buffer的E8M0量化系数Tile。每个量化系数对应K方向连续32个尾数元素，K必须为64的倍数。MXFP8支持DT_FP8E4M3FN/DT_FP8E5M2，MXFP4支持DT_FP4E2M1/DT_FP4E1M2；完整参数约束、量化系数Tensor布局和调用示例参见[matmul_mx](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx.md)和[matmul_mx_acc](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx_acc.md)。

### K维分块累加

当K维度较大，无法一次装入L1 Buffer和L0 Buffer时，需要将K轴切分为多个分块，逐块累加。首块用pypto_pro.language.matmul写入累加器，其余块用[pypto_pro.language.matmul_acc](../../../../api/pro_api/SIMD-API/cube_computation/matmul_acc.md)累加到同一个L0C Buffer。

```python
import pypto_pro.language as pl


TILE = 128
K_SIZE = 384


@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, K_SIZE, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br)
            else:
                pl.matmul_acc(ac, ac, al, br)
        pl.store(c, ac, [0, 0])
```

## 尾块处理

当GM Tensor的shape不能被Tile shape整除时，边界上会出现比Tile小的尾块。Cube场景需要为参与当前矩阵乘的输入Tile和输出Tile设置相互匹配的有效形状：

- `valid_shape`：省略时默认采用动态模式，等同于`[-1, -1]`；后续通过pypto_pro.language.set_validshape设置实际有效区域。
- `compact=1`：使L0A Buffer、L0B Buffer和L0C Buffer中的数据按有效形状采用紧凑排布，但不改变Tile绑定的Buffer大小。

```python
import pypto_pro.language as pl


tt_a_l1 = pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat)
tt_b_l1 = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat)
tt_left = pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
                      target_memory=pl.MemorySpace.Left,
                      compact=1)
tt_right = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Right,
                       compact=1)
tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Acc,
                     compact=1)
```

下面以K维完整、M和N方向存在尾块为例，运行时需要在相应的load、move和matmul执行前，同时设置L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer上Tile的有效形状：

```python
import pypto_pro.language as pl


valid_m = pl.min(TILE_M, M - i * TILE_M)
valid_n = pl.min(TILE_N, N - j * TILE_N)
pl.set_validshape(cur_a, [valid_m, TILE_K])
pl.set_validshape(cur_b, [TILE_K, valid_n])
pl.set_validshape(cur_a_left, [valid_m, TILE_K])
pl.set_validshape(cur_b_right, [TILE_K, valid_n])
pl.set_validshape(cur_acc, [valid_m, valid_n])
```

## 完整示例

以下是一个完整的Matmul Kernel，计算`C[M, N] = A[M, K] @ B[K, N]`。M和N使用动态shape，运行时传入的M、N需要分别被TILE_M、TILE_N整除；非整除场景参考上文[尾块处理](#尾块处理)。Kernel使用make_tile_group搭配auto_mutex=True，自动管理流水的同步。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 128
TILE_K = 128
TILE_N = 128
M_SIZE = 8192
K_SIZE = 128
N_SIZE = 8192


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
    b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    M = a.shape[0]
    N = b.shape[1]

    # L1 Buffer双缓冲（next()轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A Buffer/L0B Buffer/L0C Buffer单缓冲（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(core_id, M // TILE_M, num_cores):
            for j in pl.range(0, N // TILE_N, 1):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                pl.load_tile(cur_a, a, [i, 0])
                pl.load_tile(cur_b, b, [0, j])

                cur_a_left = a_left.current()
                cur_b_right = b_right.current()
                pl.move(cur_a_left, cur_a)
                pl.move(cur_b_right, cur_b)

                acc_tile = acc.current()
                pl.matmul(acc_tile, cur_a_left, cur_b_right)
                pl.store_tile(out, acc_tile, [i, j])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(42)
a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

matmul_kernel(a, b, out)
torch.npu.synchronize()

golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul kernel passed!")
```
