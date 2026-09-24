# SIMT计算

SIMT计算以线程为基本执行单元，适合表达运行时索引、逐线程分支和共享地址的原子更新。PyPTO Pro支持SIMD与SIMT混合编程：外层JIT Kernel管理Tile、数据搬运和流水依赖，SIMT函数执行逐线程计算。

初次运行可先参考[Softmax算子快速入门（SIMT）](../../../../quick_start/pro/softmax_simt.md)；概念见[SIMT编程范式](../../programming_paradigm/SIMT/programming_paradigm.md)，完整接口约束见[SIMT API](../../../../../api/pro_api/SIMT-API/index.md)。

## 定义与启动SIMT函数

### 定义入口函数和辅助函数

入口函数使用@pypto_pro.language.vector_function(mode="simt", max_threads=N)定义，每个线程执行一次函数体，结果通过Tensor或Tile写回。辅助函数使用@pypto_pro.language.vector_function(mode="simt")定义，在当前线程中执行，可以返回一个Scalar。

以下示例计算1000个FP32元素的output = input_tensor * scale + bias。辅助函数复用逐线程计算，入口函数通过Block和Thread索引定位元素，并检查尾部边界。

```python
import pypto_pro.language as pl


ELEMENTS = 1000
THREADS = 256


@pl.vector_function(mode="simt")
def affine(value: pl.DT_FP32, scale: pl.DT_FP32, bias: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + bias


@pl.vector_function(mode="simt", max_threads=THREADS)
def transform(
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    input_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if index < count:
        output[0, index] = affine(input_tensor[0, index], scale, bias)
```

参数类型可由实参推导；Tensor和Scalar参数可以写出注解，以便阅读和类型校验，Tile参数不支持使用注解。

### 在外层Vector执行域调用线程块

外层Kernel的vector section中通过`simt_func[threads](...)`调用SIMT入口函数。

```python
import pypto_pro.language as pl

@pl.jit(arch="a5")
def transform_kernel(
    input_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        transform[THREADS](output, input_tensor, count, scale, bias)
```

max_threads声明单个线程块的上限，threads指定本次调用的实际尺寸，圆括号内按位置传入实参。入口函数不能在SIMT函数中嵌套调用。

### 从Host启动外层Kernel

在A5环境中，通过Host启动外层jit kernel函数：

```python
import torch
import torch_npu


torch.npu.set_device(0)
input_tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="npu:0").reshape(1, ELEMENTS)
output = torch.empty_like(input_tensor)
scale = 2.0
bias = 1.0
blocks = (ELEMENTS + THREADS - 1) // THREADS

transform_kernel[None, blocks](input_tensor, output, ELEMENTS, scale, bias)
torch.npu.synchronize()
torch.testing.assert_close(output, input_tensor * scale + bias, rtol=0, atol=0)
```

本例在Host侧将`block_dim`设置为4，请求启动4个逻辑Vector核。假设4个核均实际生效，每个Vector核调用一次`transform[THREADS](...)`，各启动一个包含256个线程的Thread Block；最后一个Thread Block只有232个线程访问数据，其余24个线程被边界判断跳过。

Host启动参数`block_dim`用于配置逻辑核数；SIMT函数内的`pypto_pro.language.simt.block_dim()`表示Thread Block在各维度上的线程数，两者含义不同。Host启动参数的调用形式和默认值参见[Kernel核函数](../kernel_function.md#blockdim的含义与设置)。

## 配置线程与映射数据索引

方括号中的threads可以写成一至三个整数表达式，例如`simt_func[256](...)`、`simt_func[16, 16](...)`、`simt_func[8, 8, 4](...)`；pypto_pro.language.simt.block_dim()返回相应的三维尺寸，未给出的维度补1。各维必须为编译期正整数，乘积不得超过max_threads，且不得超过2048。例如`simt_func[8, 8, 4](...)`表示每个线程块有256个线程。

一维线程块可使用上例中的全局索引。二维或三维线程块可以使用X维优先的[pypto_pro.language.simt.linear_thread_idx](../../../../../api/pro_api/SIMT-API/execution/linear_thread_idx.md)展开，再结合线程块编号和每块线程总数计算全局索引：

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def copy_3d(
    source: pl.Tensor[[1, 1024], pl.DT_FP32],
    output: pl.Tensor[[1, 1024], pl.DT_FP32],
):
    dims = pl.simt.block_dim()
    threads_per_block = dims.x * dims.y * dims.z
    index = pl.simt.block_idx().x * threads_per_block + pl.simt.linear_thread_idx()
    if index < 1024:
        output[0, index] = source[0, index]
```

若外层Kernel通过`copy_3d[8, 8, 4](...)`调用该SIMT入口函数，pypto_pro.language.simt.linear_thread_idx()给出块内0至255的编号；pypto_pro.language.simt.block_idx().x乘以256后提供块偏移。例如，当实际有4个线程块时，它们分别处理索引0至255、256至511、512至767和768至1023。

## 访问Scalar、GM Tensor与UB Tile

| 对象 | 访问方式 | 使用要求 |
|---|---|---|
| Scalar | 局部变量、算术、比较、布尔表达式及pypto_pro.language.simt标量接口 | 每个线程独立计算；数学函数的多个操作数需具有相同dtype。 |
| GM Tensor | tensor[i, j]等完整标量下标 | 非零Rank、静态Shape、ND布局；索引可在运行时计算，程序须保证索引有效。 |
| UB Tile | tile[row, col] | 静态二维Shape、UB、ND布局；由外层Kernel创建和管理。 |
| Tile有效区域 | tile.valid_shape[0]、tile.valid_shape[1] | 读取运行期有效行列数，访问尾块时按有效范围保护下标。 |

Tensor/Tile须以完整对象传入，不支持元素、Slice或Tile Subview作为函数参数。元素位宽不得小于8 bit；具体Scalar计算和原子操作仍须满足各自的dtype矩阵。运行期索引不等于动态Tensor Shape，后者当前不支持。

### 访问Tile的有效区域

以下入口处理形状为[2, 128]的Tile，调用方使用`add_valid[128, 2](...)`。线程坐标覆盖物理Shape，实际读写范围由valid_shape控制：

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def add_valid(data, delta: pl.DT_FP32):
    row = pl.simt.thread_idx().y
    col = pl.simt.thread_idx().x
    if row < data.valid_shape[0] and col < data.valid_shape[1]:
        data[row, col] = data[row, col] + delta
```

有效区域由外层Kernel设置并传入。SIMT入口函数调用不会自动为Tensor/Tile元素读写添加边界判断。SIMT函数也不能创建Tile或调用pypto_pro.language.load、pypto_pro.language.store等块级搬运接口。

### 标量计算与类型转换

SIMT函数可以使用公共Scalar表达式，还提供[标量计算与类型转换API](../../../../../api/pro_api/SIMT-API/scalar_compute/index.md)，支持以下功能：

- 类型转换：pypto_pro.language.simt.cast用于数值转换，pypto_pro.language.simt.bitcast用于重新解释二进制位模式。
- 基础数学运算：pypto_pro.language.simt.abs、pypto_pro.language.simt.min、pypto_pro.language.simt.max、pypto_pro.language.simt.sqrt、pypto_pro.language.simt.rsqrt和pypto_pro.language.simt.fma分别提供绝对值、最值、平方根、平方根倒数和融合乘加。
- 指数、对数和三角函数：包括pypto_pro.language.simt.exp、pypto_pro.language.simt.exp2、pypto_pro.language.simt.log、pypto_pro.language.simt.log2、pypto_pro.language.simt.log1p、pypto_pro.language.simt.sin、pypto_pro.language.simt.cos和pypto_pro.language.simt.tanh。
- 取整运算：pypto_pro.language.simt.rint、pypto_pro.language.simt.round、pypto_pro.language.simt.floor、pypto_pro.language.simt.ceil和pypto_pro.language.simt.trunc支持按不同规则取整。
- 浮点数值判断：pypto_pro.language.simt.isnan和pypto_pro.language.simt.isinf分别判断数值是否为NaN或无穷。

## 处理数据依赖

### 外层流水同步

SIMT入口函数调用在V流水异步执行。混合计算中，MTE2搬入的数据需要就绪后才能被SIMT访问；SIMT更新的UB数据需要计算完成后才能被MTE3搬出：

```text
load（MTE2） → MTE2/V同步 → SIMD或SIMT计算（V）→ V/MTE3同步 → store（MTE3）
```

普通Tile通过成对的pypto_pro.language.system.sync_src和pypto_pro.language.system.sync_dst表达依赖，完整代码见[pypto_pro.language.system.sync_src调用示例](../../../../../api/pro_api/SIMD-API/synchronization/sync_src.md#调用示例)。

使用带mutex_ids的pypto_pro.language.make_tile_group时，默认启用的Auto Mutex可管理pypto_pro.language.load、SIMT入口函数调用和pypto_pro.language.store的缓冲区依赖。仅开启auto_mutex=True不会为普通pypto_pro.language.make_tile自动补全同步。

### 原子更新

多个线程操作同一元素时使用原子操作系列接口，原子操作不提供跨Block屏障，也不保证其他地址的数据已经就绪。完整类型和返回值规则见[原子操作API](../../../../../api/pro_api/SIMT-API/atomic/index.md)。

## 示例：用标量索引实现Gather

以下示例从输入Tensor中按indices指定的行号读取数据：

```text
output[row, col] = input_tensor[indices[0, row], col]
```

本例通过Tensor元素访问表达Gather语义。每个线程处理一个或多个输出行，行内使用循环复制；这是索引表达示例，实际性能还需结合数据布局和线程映射测量。

```python
import pypto_pro.language as pl


INPUT_ROWS = 100000
WIDTH = 128
OUTPUT_ROWS = 12288
THREADS = 256
BLOCKS = (OUTPUT_ROWS + THREADS - 1) // THREADS


@pl.vector_function(mode="simt", max_threads=THREADS)
def gather_rows(
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    row_count: pl.DT_UINT32,
):
    first_row = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    row_stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for row in pl.range(first_row, row_count, row_stride):
        input_row = indices[0, row]
        for col in pl.range(0, WIDTH, 1):
            output[row, col] = input_tensor[input_row, col]


@pl.jit(arch="a5")
def gather_kernel(
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    row_count: pl.DT_UINT32,
):
    with pl.section_vector():
        gather_rows[THREADS](output, input_tensor, indices, row_count)
```

调用方须保证0 <= row_count <= OUTPUT_ROWS，且被读取的indices元素均处于[0, INPUT_ROWS)。形状固定，索引值可以在运行时变化。准备好满足注解的NPU Tensor后，从Host启动：

```python
gather_kernel[None, BLOCKS](input_tensor, indices, output, OUTPUT_ROWS)
```

本例在Host侧将`block_dim`设置为`BLOCKS=48`，请求启动48个逻辑Vector核；每个实际生效的Vector核启动一个包含256个线程的Thread Block。线程按`pypto_pro.language.simt.grid_dim().x * pypto_pro.language.simt.block_dim().x`跨步处理后续行，覆盖全部12288行。各线程写入不同输出行，行间没有数据依赖，不需要线程块屏障。

## 当前能力边界

- SIMT入口必须由外层A5 Vector执行域调用，不支持Host直接启动SIMT函数或在SIMT函数中嵌套调用SIMT入口函数。
- SIMT中不支持Tile创建、SIMD Tile计算、Reg计算或System流水操作。
- 不支持动态GM Shape、Tile Subview、L1 Buffer Tile、DN/NZ布局和通用指针参数。
- 未提供Warp shuffle/vote/reduce、线程私有数组和显式Cached GM访问接口。
