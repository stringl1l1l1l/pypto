# Kernel核函数

Kernel是在AI Core上执行的函数，在函数签名中定义输入、输出和运行时参数，在函数体中组织数据搬运与计算。

## 定义Kernel

### 使用JIT装饰器

Kernel函数必须使用`@pypto_pro.language.jit()`装饰：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    y: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    ...
```

jit在首次启动Kernel时解析函数体并触发编译。JIT流程、编译签名和编译选项参考[JIT编译](compilation_and_execution/JIT_compilation.md)。

### 声明Kernel参数

Kernel参数需要通过类型标注明确数据类型和传递方式：

| 参数类型 | 用途 |
| --- | --- |
| [pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md) | 接收GM中的多维数据；在类型中声明shape、dtype和可选layout。 |
| [pypto_pro.language.Ptr](../../../../api/pro_api/SIMD-API/basic_data_structures/Ptr.md) | 接收裸指针；通常使用[pypto_pro.language.make_tensor](../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md)构造Tensor视图。 |
| [pypto_pro.language.DT_*](../../../../api/pro_api/SIMD-API/basic_data_structures/DataType.md) | 接收整型、浮点型等运行时标量。 |
| [TilingData类](tiling/tiling_parameter_definition.md#tilingdata) | 接收shape、stride、循环边界等结构化运行时参数。 |

Tensor参数适合直接使用调用侧Tensor的shape；Ptr参数适合由TilingData提供shape和stride：

```python
from dataclasses import dataclass

import pypto_pro.language as pl


@dataclass
class AddTiling:
    rows: int
    cols: int


@pl.jit(auto_mutex=True)
def dynamic_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
    scale: pl.DT_FP32,
    tiling: AddTiling,
):
    tensor_x = pl.make_tensor(x, [tiling.rows, tiling.cols])
    tensor_out = pl.make_tensor(out, [tiling.rows, tiling.cols])
    ...
```

TilingData必须位于Kernel形参列表和启动实参列表的末尾。完整字段和传输规则参考[Tiling参数定义与传递](tiling/tiling_parameter_definition.md#tilingdata)。

Kernel不返回Python值。计算结果通过Tensor或Ptr对应的GM区域写回。

### 定义执行域

使用[pypto_pro.language.section_vector()](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md)和[pypto_pro.language.section_cube()](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md)定义计算代码所在的执行域：

| Kernel组成 | 执行方式 |
| --- | --- |
| 仅包含Vector执行域 | 启动Vector Kernel。 |
| 仅包含Cube执行域 | 启动Cube Kernel。 |
| 同时包含Cube和Vector执行域 | 启动混合Kernel。 |

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mixed_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
):
    with pl.section_cube():
        # 矩阵计算代码
        ...

    with pl.section_vector():
        # 矢量计算代码
        ...
```

执行域决定可使用的指令、片上Buffer以及block_dim的含义。Vector计算参考[Tile计算](vector_computation/tile_computation.md)和[Reg计算](vector_computation/reg_computation.md)，矩阵计算参考[Cube计算](cube_computation.md)。

### 组织Kernel函数体

Kernel函数体通常按照以下顺序组织：

1. 定义Tile类型并绑定片上Buffer。
2. 进入Vector或Cube执行域。
3. 根据逻辑Block索引划分当前核的任务。
4. 将数据从GM搬入片上Buffer。
5. 执行矢量或矩阵计算。
6. 将结果写回GM。

下面只展示Kernel结构，Tile创建和计算参数由相应章节说明：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    y: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_x = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2])

    with pl.section_vector():
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_out = tile_out.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_out, cur_x, cur_y)
        pl.store(out, cur_out, [0, 0])
```

Tile声明、地址和TileGroup操作参考[Tile创建和操作](tile_creation_and_operations.md)。`auto_mutex=True`会将通过[pypto_pro.language.make_tile_group](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)创建的Tile对象自动插入mutex同步，确保核内Tile的读写时序；需要显式同步的场景参考[同步API](../../../../api/pro_api/SIMD-API/synchronization/index.md)。

## 调用Kernel

Kernel使用方括号指定启动配置，使用圆括号传入函数实参：

| 调用形式 | 含义 |
| --- | --- |
| kernel(args...) | 使用当前Stream，block_dim=1。 |
| kernel\[block_dim\](args...) | 使用当前Stream，并指定逻辑核数。 |
| kernel\[stream, block_dim\](args...) | 指定Stream和逻辑核数。 |
| kernel\[stream, block_dim, tiling_key\](args...) | 选择TilingKey对应的编译实例。 |
| kernel\[stream, block_dim, tiling_key, datatype\](args...) | 同时选择TilingKey和datatype特化实例。 |

使用TilingKey或datatype特化时，必须通过方括号传入相应字典。
仅使用datatype特化时，datatype字典位于第三项；同时使用TilingKey和datatype时，两者分别位于第三项和第四项。

```python
import os

import torch
import torch_npu


device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

x = torch.rand(64, 64, device=device, dtype=torch.float16)
y = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty_like(x)

add_kernel[None, 1](x, y, out)
torch.npu.synchronize()
```

Kernel启动相对于Host异步执行。在Host读取结果、检查精度或统计完整执行时间之前，需要同步对应的Stream。

### stream的含义与设置

stream指定Kernel下发的NPU执行流，用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序在device上执行。传入None时使用当前Stream；显式传入Stream时，可以只等待该Stream上的任务：

```python
import torch
import torch_npu


stream = torch.npu.Stream()
add_kernel[stream, num_cores](x, y, out)
stream.synchronize()
```

同一Stream中的任务按照下发顺序执行。不同Stream之间存在数据依赖时，需要通过Stream同步机制显式建立依赖。

### blockDim的含义与设置

block_dim表示请求启动的逻辑核数，取值必须是正整数，且不得超过当前设备和Kernel类型下实际可用的上限。运行时实际生效的逻辑Block数可能小于请求值；Kernel内应通过[pypto_pro.language.get_block_num()](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md)读取实际生效值。

以下示例假设所述核数均在当前可用范围内。

| Kernel类型 | block_dim的含义 |
| --- | --- |
| Vector Kernel | 用于设置逻辑Vector（AIV）核数。比如某款AI处理器上有40个Vector核，设置`block_dim=40`表示请求启动40个逻辑Vector核。 |
| Cube Kernel | 用于设置逻辑Cube（AIC）核数。比如某款AI处理器上有20个Cube核，设置`block_dim=20`表示请求启动20个逻辑Cube核。 |
| Cube与Vector混合Kernel | 按照AIV和AIC组合启动，用于设置逻辑执行组数。比如某款AI处理器上有40个Vector核和20个Cube核，一个执行组包含2个Vector核和1个Cube核，设置`block_dim=20`表示请求启动20个执行组；若20组均实际生效，对应40个Vector核和20个Cube核。实际生效的执行组数由`get_block_num()`获取，Vector侧逻辑核数还需乘以[pypto_pro.language.get_subblock_num()](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_num.md)。 |

## 使用TilingKey和datatype

TilingKey用于选择有限的编译期模式，datatype用于根据输入或输出数据类型生成专用实例：

```python
import pypto_pro.language as pl


key = {"UseScale": 1, "BlockM": 128}
datatype = {"x": pl.DT_FP16, "out": pl.DT_FP16}

kernel[None, block_dim, key, datatype](x, out, tiling)
```

TilingKey的声明、编码和启动规则参考[Tiling参数定义与传递](tiling/tiling_parameter_definition.md#tilingkey)。datatype字段由`@pypto_pro.language.jit(datatype=...)`声明，具体编译行为参考[JIT编译](compilation_and_execution/JIT_compilation.md#编译签名与复用)。
