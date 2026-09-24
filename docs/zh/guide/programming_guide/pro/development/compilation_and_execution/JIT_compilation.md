# JIT编译

使用`@pypto_pro.language.jit()`声明的Kernel在首次启动时完成解析、代码生成和编译，不需要单独执行编译命令。JIT适合Kernel开发、功能验证和性能调试。

## JIT编译流程

首次启动Kernel时依次执行以下步骤：

1. 绑定Kernel实参和启动配置，确定动态参数与编译期特化信息。
2. 解析Kernel函数体，生成PyPTO IR并进行校验。
3. 对IR进行必要转换，生成Device代码和Host侧Launcher。
4. 编译并加载产物，然后向指定Stream下发Kernel。

编译和执行都由一次Kernel调用触发。Kernel启动相对于Host异步，但首次调用会先等待当前编译实例生成完成。

## 触发JIT编译

Kernel首次通过`kernel[stream, block_dim](...)`启动时触发JIT编译。以下以已定义的`add_kernel`为例，展示首次启动和同一编译签名下的复用：

```python
import torch
import torch_npu


x = torch.rand(64, 64, device="npu:0", dtype=torch.float16)
y = torch.rand_like(x)
out = torch.empty_like(x)

# 首次启动：编译并执行。
add_kernel[None, 1](x, y, out)
torch.npu.synchronize()

# 编译签名相同时复用当前进程中的编译结果。
add_kernel[None, 1](x, y, out)
```

Kernel的定义和启动语法参考[Kernel核函数](../kernel_function.md)。

### 查询可用核数

设置`block_dim`前，建议使用PyTorch NPU接口查询目标Stream当前可用的最大核数：

```python
import torch
import torch_npu


stream = torch.npu.current_stream()
limits = torch.npu.get_stream_limit(stream)
max_vector_blocks = limits["vector_core_num"]
max_cube_blocks = limits["cube_core_num"]
max_mixed_blocks = min(max_cube_blocks, max_vector_blocks // 2)
```

纯Vector、纯Cube和AIC:AIV为1:2的混合Kernel分别以对应的`max_*_blocks`作为`block_dim`上限，再根据任务数量选择逻辑核数。若只需查询Device级限制，可使用`torch.npu.get_device_limit(torch.npu.current_device())`；指定Stream启动时，以目标Stream的查询结果为准。`block_dim`的含义和调用形式参见[Kernel核函数](../kernel_function.md#blockdim的含义与设置)。

## 编译签名与复用

同一Kernel对象按照编译签名区分编译实例。以下信息可能产生不同实例：

| 信息 | 对编译实例的影响 |
| --- | --- |
| Tensor固定维度 | 声明为固定值的维度在Kernel定义时编入IR，调用时Tensor的对应维度必须匹配。 |
| pypto_pro.language.STATIC维度 | 运行时取值参与特化，值变化时生成新实例。 |
| pypto_pro.language.DYNAMIC维度 | 维度值不参与特化，值变化时复用实例。 |
| Scalar入参（DT_*） | 数据类型由Kernel参数标注确定，实参值在运行时传入；值变化时复用实例。 |
| TilingKey | 每个合法Key对应一个专用实例。 |
| datatype | 每组数据类型组合对应一个专用实例。 |
| 编译目标 | 显式指定的目标在Kernel对象创建时确定；未指定时在首次启动时根据运行环境确定。不同目标使用不同的Kernel对象。 |

静态与动态shape的声明方式参考[Tensor创建和操作](../tensor_creation_and_operations.md)。Scalar入参和TilingData字段都是运行时数据，它们的值变化不会单独产生编译实例。TilingData和TilingKey的区别参考[Tiling参数定义与传递](../tiling/tiling_parameter_definition.md)。

`stream`和`block_dim`只影响本次启动，不参与编译签名；调整两者不会因此生成新的编译实例。

JIT复用范围限于当前Python进程。重新启动进程后会重新执行生成和编译流程；`build`目录中的文件用于加载和调试，不作为跨进程持久化缓存。

## JIT装饰器参数

`pypto_pro.language.jit`支持以下可配置参数：

| 参数 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| arch | str | 指定编译目标。通常省略，由运行环境自动确定。 | None |
| auto_mutex | bool | 是否根据TileGroup声明的mutex元数据，为框架能够识别的数据依赖自动插入同步。 | True |
| name | str | 自定义Kernel名称，用于区分编译产物；未设置时使用被装饰函数的名称。 | None |
| pipeline | pypto_pro.language.pipeline.PipelineConfig | 配置自动CV并行流水变换。 | None |
| sanitizer | bool | 开启Kernel内存检测，检测GM访问越界、Tile访问越界、Tile内存区间重叠和mutex未正确配对，检测结果写入报告文件，详见[内存检测](../../debug/sanitizer.md)。 | False |
| tiling_key | TilingKey定义类 | 绑定当前Kernel支持的TilingKey字段及其候选值。 | None |
| datatype | dict[str, str] | 声明需要进行数据类型特化的Kernel参数，以及Kernel函数体中引用的数据类型变量名。 | None |
| compile_timeout | int | 设置当前Kernel的编译超时时间，单位为秒。 | None（基础默认值为600秒） |

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, compile_timeout=1200, name="add_kernel")
def add_kernel(x, y, out):
    ...
```

`auto_mutex=True`只处理框架能够通过TileGroup识别的数据依赖，不能替代所有显式同步。未使用TileGroup mutex元数据，或者依赖关系无法由框架识别时，需要根据数据流显式同步。

`tiling_key`和`datatype`用于生成编译期特化实例，不会作为Kernel函数的形参传入。启动Kernel时，需要在方括号中提供对应的TilingKey和datatype字典。两者的定义和启动参数位置参考[Kernel核函数](../kernel_function.md#使用tilingkey和datatype)，TilingKey字段规则参考[Tiling参数定义与传递](../tiling/tiling_parameter_definition.md#tilingkey)。

`pipeline`接收`pypto_pro.language.pipeline.PipelineConfig`对象，用于配置自动CV并行流水变换；未设置时不执行该变换。配置方法和使用约束参考[自动CV并行流水](../../advanced_programming/auto_parallel_pipeline.md)。

## Kernel下发与同步

编译完成后，JIT通过Host侧Launcher将Kernel提交到指定Stream。`kernel[None, block_dim](...)`使用当前Stream，`kernel[stream, block_dim](...)`使用显式Stream。

Kernel下发是异步操作。在读取输出、检查精度或统计完整执行时间前，同步相应Stream：

```python
import torch
import torch_npu


stream = torch.npu.Stream()
add_kernel[stream, num_cores](x, y, out)
stream.synchronize()
```

Stream和`block_dim`的完整说明参考[Kernel核函数](../kernel_function.md#调用kernel)。

## 编译产物

未设置`ASCEND_WORK_PATH`时，JIT产物以`./build/`为根目录；设置后，以`${ASCEND_WORK_PATH}/PYPTO_PRO/build/`为根目录。Kernel目录名称以`{kernel_name}__{arch}`开头；存在STATIC维度特化时追加`__shape_{hash}`；还会根据环境附加设备和分布式进程标识，例如`__d0`、`__r1`或`__d0_r1`。设备标识是当前NPU设备编号，Rank是分布式进程编号，不是Tensor的维度数。datatype和TilingKey实例还会分别使用`dt_{hash}`和`tk_{packed}`（未使用TilingKey时为`tk_none`）子目录。主要文件包括：

| 文件 | 作用 |
| --- | --- |
| kernel.cpp | CodeGen生成的Device侧源码。 |
| call_kernel.cpp | Host侧Launcher源码，负责参数打包和Kernel下发。 |
| call_kernel_{hash}.so | 编译后的Launcher共享库。 |
| *_tiling.h | 使用TilingData时生成的结构体头文件。 |

编译失败时，优先结合错误日志和`kernel.cpp`定位解析、代码生成或工具链问题。编译成功后，中间源码默认保留在产物目录中，可用于核对生成代码。
