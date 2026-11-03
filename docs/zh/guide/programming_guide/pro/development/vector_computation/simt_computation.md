# SIMT计算

SIMT计算以线程为基本执行单元，适合表达运行时索引、逐线程分支和共享地址的原子更新。

PyPTO Pro支持Host直接启动SIMT kernel，也支持在外层Kernel中调用SIMT vector function（简称SIMT VF），组合SIMD与SIMT计算。线程模型、数据对象和两种编程方式的关系参见[SIMT编程范式](../../programming_paradigm/SIMT/programming_paradigm.md)，完整接口约束见[SIMT API](../../../../../api/pro_api/SIMT-API/index.md)。

## SIMT编程

### 定义SIMT kernel

SIMT kernel使用@pypto_pro.language.jit(mode="simt", max_threads=N)定义，直接描述每个线程执行的函数体，无需包裹Vector执行域。

max_threads声明每个Thread Block的线程数上限，须为[1, 2048]范围内的整数，不接受bool。实际线程配置在Host启动时传入，各维度的线程数乘积不能超过max_threads。该参数还参与编译期寄存器资源分配，应结合单线程计算复杂度选择，参见[配置最大线程数](../../programming_paradigm/SIMT/programming_paradigm.md#配置最大线程数)。

SIMT kernel的定义须满足以下约束：

- Kernel形参须提供类型注解，仅接受Scalar和Tensor。
- 装饰器入参不支持pipeline、tiling_key、datatype特化和sanitizer=True。SIMT kernel不使用Auto Mutex，线程间同步由开发者通过SIMT同步接口组织。

### 从Host启动SIMT kernel

Host通过以下两种形式启动SIMT kernel：

- `kernel[grid, block](...)`：使用当前NPU执行流。
- `kernel[stream, grid, block](...)`：指定执行流，stream为None时使用当前执行流。

grid指定线程块数量，block指定每个线程块的线程数。SIMT kernel必须通过方括号指定grid和block，不支持直接使用`kernel(...)`启动。

| 参数 | 配置形式 | 约束 |
|---|---|---|
| grid | 正整数，或包含一至三个正整数的tuple/list | 各维度乘积不超过65535。 |
| block | 正整数，或包含一至三个正整数的tuple/list | 各维度乘积不超过定义时的max_threads。 |

各维未给出的分量补1，不接受bool。例如，`kernel[4, 256](...)`启动4个Thread Block，每块256个线程；`kernel[(2, 3, 4), (8, 4, 2)](...)`启动24个Thread Block，每块64个线程。grid和block属于本次启动配置，改变它们不会单独生成新的编译实例。

定义与调用示意如下，省略逐线程计算逻辑。data为Host侧已准备的Shape为[1, 1024]的FP32 NPU Tensor：

```python
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


@pl.jit(mode="simt", max_threads=256)
def simt_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    ...


simt_kernel[4, 256](data)
```

## SIMD与SIMT混合编程

### 定义SIMT VF

SIMT VF使用@pypto_pro.language.vector_function(mode="simt", max_threads=N)定义，每个线程执行一次函数体，结果通过传入的Tensor或Tile写回。

max_threads的约束与[SIMT kernel](#定义simt-kernel)一致，实际线程数由SIMT VF调用点指定。

- SIMT VF可接受Scalar、Tensor和Tile作为入参，参数类型可由实参推导；Tensor和Scalar参数可以写出注解，以便阅读和类型校验，Tile参数不支持使用注解。

### 调用SIMT VF

外层Kernel使用@pypto_pro.language.jit()定义，在pypto_pro.language.section_vector()执行域中通过`simt_func[threads](...)`调用SIMT VF。方括号内指定本次调用的Thread Block尺寸，圆括号内按位置传入实参。

| 线程块配置 | 调用形式 | 每块线程数 |
|---|---|---|
| 一维 | `simt_func[256](...)` | 256 |
| 二维 | `simt_func[16, 16](...)` | 256 |
| 三维 | `simt_func[8, 8, 4](...)` | 256 |

各维须为编译期正整数表达式，未给出的分量补1，各维度乘积不超过SIMT VF定义时的max_threads。线程数取32的整数倍可减少最后一个Warp的空闲线程通道，但不是必须条件。

每个执行到调用点的Vector核启动一个Thread Block；Grid由实际生效的Vector核数决定，仅使用X维，Y、Z维均为1，无需在SIMT VF调用点单独配置grid。

SIMT VF在V流水异步执行，与数据搬运之间的依赖须由外层Kernel组织，参见[外层流水同步](#外层流水同步)。

### 从Host启动外层Kernel

Host通过`kernel[block_dim](...)`或`kernel[stream, block_dim](...)`启动外层Kernel，SIMT VF随外层Kernel执行到调用点时启动。

Host启动参数block_dim用于配置逻辑核数；SIMT函数内的pypto_pro.language.simt.block_dim()表示Thread Block在各维度上的线程数，两者含义不同。Host启动参数的默认值和实际生效核数参见[Kernel核函数](../kernel_function.md#blockdim的含义与设置)。

定义与调用示意如下，省略逐线程计算逻辑。data为Host侧已准备的Shape为[1, 1024]的FP32 NPU Tensor：

```python
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


@pl.vector_function(mode="simt", max_threads=256)
def simt_vf(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    ...


@pl.jit(arch=NpuArch.DAV_3510)
def mixed_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        simt_vf[256](data)


mixed_kernel[None, 4](data)
```

## SIMT helper

SIMT helper用于复用逐线程计算逻辑，使用@pypto_pro.language.vector_function(mode="simt")定义，不填写max_threads。函数体中可以使用Scalar表达式和SIMT接口。

- SIMT kernel、SIMT VF或其他SIMT helper通过普通函数调用使用helper，不使用方括号配置线程。helper在调用者当前线程中执行，沿用入口的线程配置，不启动新的线程块。
- helper可以不返回值或返回一个Scalar，也可以将结果写入传入的Tensor或Tile。返回的Scalar属于当前调用线程。
- helper的参数约束由调用入口决定：SIMT kernel调用链只接受Scalar和Tensor；SIMT VF调用链还可接受符合要求的UB Tile。

以下示例在SIMT kernel中调用helper，SIMT VF中的调用写法相同。data为Host侧已准备的Shape为[1, 1024]的FP32 NPU Tensor：

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt")
def add_one(value: pl.DT_FP32) -> pl.DT_FP32:
    return value + 1.0


@pl.jit(mode="simt", max_threads=256)
def update_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.linear_thread_idx()
    if index < 1024:
        data[0, index] = add_one(data[0, index])


update_kernel[4, 256](data)
```

## 数据对象

### 数据对象与内存层级

SIMT中的数据对象包括Scalar、Tensor和Tile，分别用于表达线程私有数据、GM数据及UB数据。内存的共享范围、生命周期和资源划分参见[内存层级](../../programming_paradigm/SIMT/programming_paradigm.md#内存层级)。

| 对象 | 内存对应关系 | 访问方式 | 使用要求 |
|---|---|---|---|
| Scalar | 线程私有，通常使用寄存器，也可能使用栈空间 | 局部变量、算术、比较、布尔表达式及pypto_pro.language.simt标量接口 | 每个线程独立计算；各操作数的数据类型须满足对应接口约束。 |
| Tensor | GM，全局内存 | tensor[i, j]等完整标量下标 | 非零Rank、静态Shape、ND布局；索引可在运行时计算，程序须保证索引有效。 |
| Tile | UB，共享内存 | tile[row, col] | 仅用于混合编程，静态二维Shape、UB、ND布局；由外层Kernel创建和管理。 |

Tensor和Tile元素位宽不得小于8 bit，具体Scalar计算和原子操作还须满足各接口的数据类型约束。Tensor和Tile作为函数参数时须以完整对象传入，不支持Slice或Tile Subview；元素读取所得的Scalar可用于Scalar参数。

Tensor解析后的Shape必须为静态，不支持pypto_pro.language.DYNAMIC维度；使用pypto_pro.language.STATIC特化的维度须在启动时解析为静态值。运行期索引不等于动态Tensor Shape。

SIMT kernel及其调用的SIMT helper只接受Scalar和Tensor；SIMT VF及其调用的SIMT helper还可接受UB Tile，不支持L1 Buffer Tile或DN/NZ布局。当前未提供线程私有数组和显式Cached GM访问接口。

### Tile和Tensor读写

SIMT函数均通过完整标量下标访问Tile或Tensor元素。以二维对象a为例：

| 操作 | 表达方式 | 含义 |
|---|---|---|
| 读取 | value = a[row, col] | 读取一个元素，结果为Scalar。 |
| 写入 | a[row, col] = value | 将Scalar写入一个元素。 |
| 读取后写回 | a[row, col] = a[row, col] + delta | 读取、计算后写回；不具备原子性。 |

### Tensor和Tile读写的有效区域

当前SIMT对Tensor和Tile元素读写仅校验编译期常量下标是否越界，不会自动生成运行期边界判断。

- 静态常量下标：前端能够求值为整数常量的下标，会按对应维度的Shape检查，负数或超出该维度范围的下标会报错。例如，Shape为[1, 64]时，a[0, 64]、a[1, 0]和a[0, -1]均会报错。
- 动态下标：由运行期Scalar入参、Tensor元素、线程编号或其计算表达式得到的下标，前端不会推导其取值范围，由开发者保证其读写数据的有效性，可以在读写前显式添加0 <= index < Shape[axis]条件判断。

访问Tile的有效数据还须保证下标小于对应维度的valid_shape；有效区域由外层Kernel设置并传入，SIMT不会自动按valid_shape限制读写。越界读写可能破坏相邻数据或引发执行异常。

以下入口处理形状为[2, 128]的Tile入参data，通过data.valid_shape[0]和data.valid_shape[1]分别获取该Tile运行期的有效行数和有效列数，并通过条件判断将实际读写限制在有效区域内：

```python
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch

@pl.vector_function(mode="simt", max_threads=256)
def add_valid(data, delta: pl.DT_FP32):
    row = pl.simt.thread_idx().y
    col = pl.simt.thread_idx().x
    if row < data.valid_shape[0] and col < data.valid_shape[1]:
        data[row, col] = data[row, col] + delta
```

## SIMT编程接口体系

SIMT提供以下接口，用于线程索引查询、同步及逐线程计算：

### 线程层级与索引

[执行模型API](../../../../../api/pro_api/SIMT-API/execution/index.md)用于查询本次启动的线程层级和当前线程位置：

| 接口 | 含义 |
|---|---|
| pypto_pro.language.simt.grid_dim() | Grid在各维度上的Thread Block数量。 |
| pypto_pro.language.simt.block_dim() | Thread Block在各维度上的线程数。 |
| pypto_pro.language.simt.block_idx() | 当前Thread Block在Grid中的坐标。 |
| pypto_pro.language.simt.thread_idx() | 当前线程在Thread Block中的坐标。 |
| pypto_pro.language.simt.linear_thread_idx() | 当前线程在块内按X维优先展开的编号，不含线程块偏移。 |

前四个接口返回dim3形式的三维对象，各分量为DT_UINT32 Scalar；pypto_pro.language.simt.linear_thread_idx()返回DT_UINT32 Scalar。

**SIMT编程中的线程索引**

一维Grid和一维Thread Block的全局索引等于“线程块X坐标 × 每块线程数 + 线程X坐标”。三维配置则先展平线程块坐标，再叠加块内线性线程编号：

```python
import pypto_pro.language as pl


@pl.jit(mode="simt", max_threads=64)
def copy_grid(
    source: pl.Tensor[[1536], pl.DT_FP32],
    output: pl.Tensor[[1536], pl.DT_FP32],
):
    grid = pl.simt.grid_dim()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    linear_block = block_id.x + grid.x * (block_id.y + grid.y * block_id.z)
    threads_per_block = block.x * block.y * block.z
    index = linear_block * threads_per_block + pl.simt.linear_thread_idx()
    output[index] = source[index]
```

使用`copy_grid[(2, 3, 4), (8, 4, 2)](source, output)`启动时，24个线程块各含64个线程，共覆盖1536个元素。该示例的启动范围与Tensor大小相同；处理其他尺寸时，须在元素访问前判断索引是否小于数据范围。

**混合编程中的线程索引**

SIMD与SIMT混合编程的Grid仅使用X维，线程块Y、Z坐标均为0。每个线程块仍可配置二维或三维线程，并使用pypto_pro.language.simt.linear_thread_idx()展平块内坐标：

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

外层Kernel通过`copy_3d[8, 8, 4](...)`调用时，每个线程块包含256个线程。例如，实际生效4个Vector核时，4个线程块分别处理索引0至255、256至511、512至767和768至1023。数据量超过本次启动的总线程数时，可按总线程数跨步处理，见[Gather示例](#simd与simt混合编程示例gather)。

### 同步

根据依赖发生的位置选择同步方式：

| 依赖场景 | 使用方式 |
|---|---|
| 当前线程块内，各线程完成一个阶段后再进入下一阶段 | pypto_pro.language.simt.syncthreads。 |
| 约束当前线程的内存访问顺序 | 按作用范围选择pypto_pro.language.simt.threadfence_block或pypto_pro.language.simt.threadfence。 |
| 混合编程中，数据搬运与SIMT计算之间的依赖 | 外层Kernel中的流水同步或Auto Mutex。 |

#### 线程块同步

pypto_pro.language.simt.syncthreads设置当前线程块的屏障。比如混合编程中，各线程先写入共享UB Tile，再读取其他线程写入的数据，应在写入和读取两个阶段之间设置屏障。完整示例见[pypto_pro.language.simt.syncthreads](../../../../../api/pro_api/SIMT-API/synchronization/syncthreads.md)。

同一线程块的所有线程必须到达同一个同步点。存在尾部边界判断时，应让无有效数据的线程跳过读写，并继续到达需要全块参与的屏障；不能让这些线程提前return或跳过含屏障的分支。循环中的屏障也要求各线程执行相同次数。

该接口只同步当前线程块，不提供整个Grid的屏障。SIMT helper中的屏障同样要求调用线程满足一致到达条件。

#### 内存栅栏

内存栅栏约束调用线程的内存访问顺序，不会等待其他线程到达同一位置：

- [pypto_pro.language.simt.threadfence_block](../../../../../api/pro_api/SIMT-API/synchronization/threadfence_block.md)：在线程块范围内建立内存访问顺序。
- [pypto_pro.language.simt.threadfence](../../../../../api/pro_api/SIMT-API/synchronization/threadfence.md)：在设备范围内建立内存访问顺序。

例如，线程先写数据，再通过原子变量发布完成状态时，需要考虑数据写入与完成状态更新的顺序，以及消费者如何观察完成状态。内存栅栏本身既不通知消费者，也不保证并发更新的原子性，须与完整的线程协作协议配合使用。机制与示例参见[SIMT同步简介](../../../../../api/pro_api/SIMT-API/synchronization/synchronization_intro.md)。

#### 外层流水同步

SIMD与SIMT混合编程中，SIMT VF调用在V流水异步执行。MTE2搬入的数据需要就绪后才能被SIMT访问；SIMT更新的UB数据需要计算完成后才能被MTE3搬出：

```text
load（MTE2） → MTE2/V同步 → SIMD或SIMT计算（V）→ V/MTE3同步 → store（MTE3）
```

普通Tile通过成对的pypto_pro.language.system.sync_src和pypto_pro.language.system.sync_dst表达依赖，完整代码见[pypto_pro.language.system.sync_src调用示例](../../../../../api/pro_api/SIMD-API/synchronization/sync_src.md#调用示例)。

使用带mutex_ids的pypto_pro.language.make_tile_group时，默认启用的Auto Mutex可管理pypto_pro.language.load、SIMT入口函数调用和pypto_pro.language.store的缓冲区依赖。仅开启auto_mutex=True不会为普通pypto_pro.language.make_tile自动补全同步。

### Warp内协作

需要在Warp内查询线程状态、交换数据或进行归约时，可使用[Warp API](../../../../../api/pro_api/SIMT-API/warp/index.md)。Warp接口的参与线程、来源Lane及数据类型须满足各接口约束，不能用Warp内协作代替整个Thread Block的同步。

### 数学函数与类型转换

SIMT函数可以使用公共Scalar表达式，还提供以下常用功能，此处仅列举部分接口：

- 类型转换：如pypto_pro.language.simt.cast。
- 基础数学运算：如pypto_pro.language.simt.abs、pypto_pro.language.simt.sqrt。
- 指数、对数和三角函数：如pypto_pro.language.simt.exp、pypto_pro.language.simt.log、pypto_pro.language.simt.sin。
- 取整运算：如pypto_pro.language.simt.floor。
- 浮点数值判断：如pypto_pro.language.simt.isnan。

完整接口及使用约束见[标量计算与类型转换API](../../../../../api/pro_api/SIMT-API/scalar_compute/index.md)。

### 原子操作

多个线程向同一元素累加、更新最值或修改状态时，普通的“读取、计算、写回”可能丢失其他线程的更新。例如，多个线程同时执行counter[0, 0] = counter[0, 0] + 1，不能保证最终结果等于参与线程数。

此时可使用[原子操作API](../../../../../api/pro_api/SIMT-API/atomic/index.md)。例如，counter为DT_INT32 Tensor时，可在SIMT函数体内调用`pypto_pro.language.simt.atomic_add(counter[0, 0], 1)`完成一次原子累加：

- target须直接使用Tensor或Tile的单元素下标表达式；SIMT kernel使用GM Tensor目标，混合编程也可使用UB Tile目标。
- 原子操作的数据类型、输入值和返回值须满足对应接口约束，部分类型不返回更新前的值。
- 计数器等共享状态须在并发更新开始前初始化完成，例如在Host侧准备零值Tensor。
- 原子操作保护目标元素的更新，不提供跨线程块屏障，也不保证其他地址的数据已就绪。

原子操作常用于计数、直方图和共享状态更新。大量线程竞争同一地址可能降低吞吐量，可根据算法考虑减少共享更新次数。

## 完整编程示例

### SIMT编程示例：逐线程仿射变换

以下示例处理1000个FP32元素，计算output = input_tensor × scale + bias。每个线程处理一个元素，SIMT helper复用逐线程计算逻辑：

1. 每个Thread Block配置256个线程，Host启动4个线程块。
2. 通过线程块编号和块内线程编号计算元素索引。
3. 仅在索引小于元素数量时读写Tensor，最后一个线程块中的24个线程跳过读写。
4. Host等待Device执行完成后校验结果。

```python
import pypto_pro.language as pl
import torch
import torch_npu


ELEMENTS = 1000
THREADS = 256


@pl.vector_function(mode="simt")
def affine_simt(value: pl.DT_FP32, scale: pl.DT_FP32, bias: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + bias


@pl.jit(mode="simt", max_threads=THREADS)
def transform_simt_kernel(
    output: pl.Tensor[[ELEMENTS], pl.DT_FP32],
    input_tensor: pl.Tensor[[ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.linear_thread_idx()
    if index < count:
        output[index] = affine_simt(input_tensor[index], scale, bias)


torch.npu.set_device(0)
input_tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="npu:0")
output = torch.empty_like(input_tensor)
grid = (ELEMENTS + THREADS - 1) // THREADS
transform_simt_kernel[grid, THREADS](output, input_tensor, ELEMENTS, 2.0, 1.0)
torch.npu.synchronize()
torch.testing.assert_close(output, input_tensor * 2.0 + 1.0, rtol=0, atol=0)
```

本例各线程写入不同元素，线程之间没有数据依赖，不需要线程块屏障。Tensor访问不会自动补充运行期边界判断，因此函数体保留index < count条件；调用方须保证count不超过Tensor大小。

### SIMD与SIMT混合编程示例：Gather

以下示例从输入Tensor中按indices指定的行号读取数据：

```text
output[row, col] = input_tensor[indices[0, row], col]
```

本例通过Tensor元素访问表达Gather语义。每个线程处理一个或多个输出行，行内使用循环复制；这是索引表达示例，实际性能还需结合数据布局和线程映射测量。

```python
import pypto_pro.language as pl
import torch
import torch_npu
from pypto_pro.runtime.platform import NpuArch


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


@pl.jit(arch=NpuArch.DAV_3510)
def gather_kernel(
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    row_count: pl.DT_UINT32,
):
    with pl.section_vector():
        gather_rows[THREADS](output, input_tensor, indices, row_count)


torch.npu.set_device(0)
input_tensor = torch.randn(INPUT_ROWS, WIDTH, dtype=torch.float32, device="npu:0")
indices = torch.randint(INPUT_ROWS, (1, OUTPUT_ROWS), dtype=torch.int32, device="npu:0")
output = torch.empty((OUTPUT_ROWS, WIDTH), dtype=torch.float32, device="npu:0")

gather_kernel[None, BLOCKS](input_tensor, indices, output, OUTPUT_ROWS)
torch.npu.synchronize()

expected = input_tensor.index_select(0, indices.reshape(-1).to(torch.int64))
torch.testing.assert_close(output, expected, rtol=0, atol=0)
```

调用方须保证0 <= row_count <= OUTPUT_ROWS，且被读取的indices元素均处于[0, INPUT_ROWS)。Tensor形状固定，索引值可以在运行时变化。

本例在Host侧通过BLOCKS将block_dim设置为48，请求启动48个逻辑Vector核。BLOCKS须不超过当前执行流可用的Vector核数，可根据环境调小。每个实际生效的Vector核启动一个包含256个线程的Thread Block；线程按实际Grid中的总线程数跨步处理后续行，因此调小BLOCKS后仍能覆盖全部12288行。各线程写入不同输出行，行间没有数据依赖，不需要线程块屏障。
