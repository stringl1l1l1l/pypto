# Tiling参数定义与传递

Tiling参数在Host侧计算后传给Kernel，由TilingKey和TilingData共同承担：

| 类型 | 生效阶段 | 承担的部分 |
| --- | --- | --- |
| TilingKey | Kernel编译时 | 选择本次使用哪一套实现，例如对齐与非对齐、不同的Tile模板或数据排布。候选数量有限，每个取值组合生成一个专用Kernel实例。 |
| TilingData | Kernel运行时 | 携带这一套实现所需的具体数值，例如shape、stride、循环边界和缩放系数。数值变化时复用同一份已编译的Kernel。 |

即TilingKey决定走哪个分支，TilingData提供该分支运行所需的数据。本文分别说明两者的声明、传递方式，以及配合使用的方法。Tiling的概念和切分层次参见[Tiling概述](tiling_overview.md)。

## TilingData

TilingData使用Python `dataclass`声明。框架根据字段标注生成设备侧结构体，并在启动Kernel时序列化字段值。一个Kernel最多只能声明一个TilingData参数，且该参数必须位于形参列表末尾。

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AddTiling:
    rows: int
    cols: int
    scale: float
    strides: int[2]
```

### 字段类型

TilingData的字段支持标量`int`、`float`、`bool`，以及对应的定长数组`int[N]`、`float[N]`、`bool[N]`。其中，`int`对应DT_INT64，`float`对应DT_FP32，`bool`对应DT_BOOL。

字段声明需满足以下要求：

- 类中至少包含一个字段，每个字段只能使用上述类型。
- 定长数组的长度`N`必须直接写成1～2048的整数值；运行时传入的列表长度必须与声明一致。
- 使用`int[N]`、`float[N]`或`bool[N]`时，需要启用`from __future__ import annotations`。
- 字段可以设置`dataclass`默认值；数组默认值使用`dataclasses.field(default_factory=...)`。
- 字段顺序决定设备侧结构体的字段顺序、偏移和对齐，不能在Host侧和设备侧使用不同顺序解释同一份数据。

### 在Host侧传入TilingData

Host侧构造TilingData实例，并将其放在Kernel启动实参列表末尾（下面的`add_kernel`是一个接收`AddTiling`的Kernel，定义见[在Kernel中读取TilingData](#在kernel中读取tilingdata)）：

```python
tiling = AddTiling(
    rows=513,
    cols=511,
    scale=1.0,
    strides=[511, 1],
)

block_dim = 8
add_kernel[None, block_dim](x, y, z, tiling)
```

使用默认值时，未显式传入的字段采用`dataclass`默认值：

```python
from dataclasses import dataclass, field


@dataclass
class CopyTiling:
    length: int
    scale: float = 1.0
    offsets: int[4] = field(default_factory=lambda: [0, 0, 0, 0])
```

带默认值的字段必须位于无默认值字段之后。

### 在Kernel中读取TilingData

在Kernel函数签名中声明TilingData参数后，即可通过字段名读取运行时值：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    strides = [tiling.strides[0], tiling.strides[1]]
    tensor_x = pl.make_tensor(x, [tiling.rows, tiling.cols], strides)
    tensor_y = pl.make_tensor(y, [tiling.rows, tiling.cols], strides)
    tensor_z = pl.make_tensor(z, [tiling.rows, tiling.cols], strides)

    tile_rows = (tiling.rows + 127) // 128
    tile_cols = (tiling.cols + 127) // 128

    for tile_idx in pl.range(
        pl.get_block_idx(),
        tile_rows * tile_cols,
        pl.get_block_num(),
    ):
        ...
```

TilingData字段是运行时值，可以用于：

- `pypto_pro.language.make_tensor`的shape和stride。
- `pypto_pro.language.range`的循环边界。
- Tile数量、偏移和有效形状等标量计算。
- 运行时`if`条件和算术表达式。

TilingData字段不能用于要求编译期Python常量的参数，例如`TileType`的静态`shape`。这类有限候选项应通过TilingKey选择。

## TilingKey

TilingKey描述一组有限的编译期配置。同一份Kernel源码可以针对不同Key生成多个专用实例，并在启动时选择其中一个。Key字段在Kernel中是编译期常量，对应的条件分支可以在编译阶段确定。

### 声明TilingKey

TilingKey使用普通Python类声明，类属性使用`TilingKeyField`定义候选值：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField


class AddKey:
    UseScale = TilingKeyField(bits=1, values=[0, 1])
    BlockM = TilingKeyField(bits=2, values=[64, 128])

    def is_valid(self, key):
        use_scale, block_m = key
        return use_scale == 0 or block_m == 128
```

字段按照类中的定义顺序收集。该顺序决定`is_valid()`参数元组的顺序和64-bit Key中各字段的位置。各字段候选值的下标打包为64-bit Key，是为了与离线二进制编译场景使用的`uint64_t` TilingKey配合。

`TilingKeyField`需满足以下约束：

| 项目 | 约束 |
| --- | --- |
| `bits` | 必须大于0。 |
| `values` | 候选值只能是整数，不能使用字符串、浮点数或`bool`；集合必须非空且值互不重复。 |
| 编码容量 | 候选值数量不能超过`2**bits`。 |
| 总位宽 | 所有字段的`bits`之和不能超过64。 |
| `is_valid` | 可选；用于拒绝不支持的字段组合。 |

字段编码保存候选值在`values`中的下标，而不是候选值本身。例如：

```python
class BlockKey:
    BlockM = TilingKeyField(bits=2, values=[32, 64, 128])
```

`BlockM=32`、`64`、`128`分别编码为`0`、`1`、`2`。Host侧启动Kernel时仍传入实际值，不需要计算候选下标。

### 将TilingKey绑定到Kernel

通过`@pypto_pro.language.jit(tiling_key=...)`绑定TilingKey。字段名可以直接在Kernel函数体中使用，无需加入形参列表：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, tiling_key=AddKey)
def add_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    tile_type = pl.TileType(
        shape=[BlockM, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    if UseScale == 1:
        # 此分支在编译时确定
        ...
    else:
        ...
```

TilingKey字段名不能与Kernel形参或模块级普通变量重名。字段应只描述会影响代码生成的有限模式，不要为任意运行时shape分别定义Key。

### 选择TilingKey并启动Kernel

Key字典位于启动参数中的`stream`和`block_dim`之后：

```python
key = {"UseScale": 1, "BlockM": 128}
add_kernel[None, block_dim, key](x, y, z, tiling)
```

Key字典必须满足以下要求：

- 包含Schema中的全部字段，且不能包含额外字段。
- 每个值都属于对应字段的`values`。
- 字段组合能够通过`is_valid()`校验。
- 传入字典中的值是实际候选值，不是编码后的下标。

同时使用datatype特化时，datatype字典位于TilingKey之后：

```python
add_kernel[None, block_dim, key, datatype](x, y, z, tiling)
```

二进制交付时，TilingKey Schema用于生成字段声明和合法的Key组合，具体流程参考[离线二进制编译](../compilation_and_execution/offline_binary_compilation.md)。

## TilingData与TilingKey配合使用

同一个Kernel可以使用TilingKey选择编译期实现，再通过TilingData接收本次调用的运行时参数：

```python
key = {"UseScale": 1, "BlockM": 128}
tiling = AddTiling(rows=m, cols=n, scale=0.5, strides=[n, 1])

add_kernel[None, block_dim, key](x, y, z, tiling)
```

选择参数传输方式时，可以按以下规则判断：

| 参数特点 | 传输方式 |
| --- | --- |
| 每次调用都可能变化，且不要求重新生成代码 | TilingData |
| 候选数量有限，并影响代码分支、Tile模板或排布 | TilingKey |
| 只决定启动的核数 | `block_dim`，参考[blockDim的含义与设置](../kernel_function.md#blockdim的含义与设置) |

## 使用限制与建议

- 不要把任意shape值放入TilingKey，否则候选组合会快速增加，并产生大量专用Kernel。
- 不要把影响静态Tile类型的参数放入TilingData，这类参数在Kernel运行时才可用。
- 修改TilingData字段顺序或类型会改变设备侧结构体布局，Host侧与Kernel必须使用同一定义。
- 数组字段长度必须与声明完全一致；运行时可变长度数据应拆成长度字段和固定容量数组，或通过Tensor传递。
- 同时使用TilingKey、datatype特化和TilingData时，分别检查启动方括号参数与函数调用参数的顺序。
