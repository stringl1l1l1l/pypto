# Tile创建和操作

Tile表示AI Core片上Buffer中的二维数据块，记录其shape、数据类型、内存空间、排布和绑定地址等信息，是数据搬运和计算接口的基本操作对象。

## Tile创建流程

创建Tile通常包含以下步骤：

1. 使用TileType描述Tile规格。
2. 使用make_tile创建单个Tile，或使用make_tile_group创建一组可轮转Tile。
3. 对TileGroup使用next()、current()、previous()或下标访问取得具体Tile。
4. 尾块场景根据实际有效范围设置valid_shape，需要时通过reinterpret创建共享同一缓冲区的新视图。

Tile和Tensor的作用不同：Tensor描述GM中的数据视图，Tile表示AI Core片上Buffer中的数据块。Tensor的声明和视图操作请参考[Tensor创建和操作](tensor_creation_and_operations.md)。

## 使用TileType描述Tile

pypto_pro.language.TileType只描述Tile规格，不分配片上空间。其主要参数如下：

```python
tile_type = pypto_pro.language.TileType(
    shape=[64, 128],
    dtype=pypto_pro.language.DT_FP16,
    target_memory=pypto_pro.language.MemorySpace.Vec,
    valid_shape=None,
    layout=None,
    fractal=None,
    pad=None,
    compact=None,
)
```

| 参数 | 说明 |
|:---|:---|
| shape | Tile的物理shape，当前仅支持二维正整数shape |
| dtype | Tile元素的数据类型 |
| target_memory | Tile所在的片上内存空间，默认为MemorySpace.Vec |
| valid_shape | Tile的有效区域；省略时默认动态模式，等同于[-1, -1]，尾块在运行时通过set_validshape设置 |
| layout | Tile的物理排布，如ND、DN、NZ、ZN、NN或ZZ；省略时按内存空间推导默认值 |
| fractal | 分形大小；部分内存空间和数据类型可以推导默认值 |
| pad | 无效区域的填充模式 |
| compact | Tile有效数据在缓冲区中的紧凑排布方式 |

valid_shape、pad和compact主要用于尾块及特殊排布场景。它们不会改变make_tile或make_tile_group绑定的缓冲区大小。完整约束请参考[pypto_pro.language.TileType](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)。

### 选择片上内存空间

target_memory决定Tile绑定的物理缓冲区和可使用的数据路径：

| pypto_pro.language.MemorySpace | 物理缓冲区 | 典型用途 |
|:---|:---|:---|
| Vec | UB | 矢量数据的输入、输出和中间结果 |
| Mat | L1 Buffer | GM与L0A Buffer/L0B Buffer之间的矩阵暂存 |
| Left | L0A Buffer | 矩阵乘左操作数 |
| Right | L0B Buffer | 矩阵乘右操作数 |
| Acc | L0C Buffer | 矩阵乘累加结果 |
| Bias | BiasTable Buffer | 矩阵乘的融合偏置 |
| Scaling | Fixpipe Buffer | FIX数据通路的per-channel随路量化/反量化参数 |
| ScaleLeft | L0A_MX Buffer | MX矩阵乘的左量化系数矩阵 |
| ScaleRight | L0B_MX Buffer | MX矩阵乘的右量化系数矩阵 |

选择内存空间后，还需要遵守相应Buffer对layout、fractal、dtype、容量和地址对齐的约束。详细枚举说明请参考[pypto_pro.language.MemorySpace](../../../../api/pro_api/SIMD-API/basic_data_structures/MemorySpace.md)。

## Tile创建方式的选择

确定TileType后，可以通过make_tile创建一块绑定固定地址的Tile，由开发者管理访问时序和同步；也可以通过make_tile_group创建一组规格相同的Tile，统一管理地址、轮转和可选的自动同步元数据。根据缓冲方式和同步需求选择：

| 需求 | 建议方式 |
|:---|:---|
| 创建一块固定地址的Tile并精确控制同步 | make_tile |
| 创建仅在局部计算中临时使用、无需轮转和自动同步的Tile | make_tile |
| 创建常规单缓冲并使用自动同步 | make_tile_group，配置一个mutex ID |
| 创建双缓冲或N缓冲 | make_tile_group，按缓冲深度配置多个mutex ID |
| 创建轮转Tile但自行管理同步 | make_tile_group，使用depth且不配置mutex ID |

## 使用make_tile创建单个Tile

pypto_pro.language.make_tile在TileType指定的内存空间中绑定一段固定地址：

```python
tile = pypto_pro.language.make_tile(
    tile_type,
    addr=0x0000,
)
```

- addr是目标Buffer内的字节偏移，必须在编译期确定。
- 地址范围的字节数由TileType的shape和dtype自动推导，不接受size参数；valid_shape不改变分配范围。
- 多个Tile的地址范围不能发生非预期重叠。有意复用同一地址时，开发者必须保证访问时序正确。

下面的示例仅创建三块UB Tile，不包含数据搬运和计算：

```python
import pypto_pro.language as pl

tt = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)

# 单块Tile占用64 * 128 * 2 = 16384字节，即0x4000字节。
tile_a = pl.make_tile(tt, addr=0x0000)
tile_b = pl.make_tile(tt, addr=0x4000)
tile_out = pl.make_tile(tt, addr=0x8000)
```

使用make_tile时，Tile不携带供auto_mutex使用的TileGroup轮转和mutex元数据。跨Pipe依赖需要开发者根据实际数据流显式同步。接口详情请参考[pypto_pro.language.make_tile](../../../../api/pro_api/SIMD-API/resource_management/make_tile.md)。

### 地址对齐

不同Buffer对Tile起始地址有不同的基本对齐要求：

| Buffer | addr对齐要求 |
|:---|:---|
| UB、L1 Buffer | 32字节 |
| L0A Buffer、L0B Buffer | 512字节 |
| L0C Buffer | 64字节 |
| L0A_MX Buffer、L0B_MX Buffer | 32字节 |

除基本地址对齐外，Tile实际占用空间还会受到dtype、layout、fractal和shape对齐方式的影响。对于NZ、ZN等分形排布，不能只按逻辑元素数量估算地址范围。

L0A_MX Buffer和L0B_MX Buffer中的量化系数Tile地址还必须分别等于配套L0A Buffer和L0B Buffer Tile地址右移4位。

## 使用make_tile_group创建轮转Tile

pypto_pro.language.make_tile_group一次创建一组规格相同的Tile，并统一管理地址排布、轮转位置和mutex元数据：

```python
group = pypto_pro.language.make_tile_group(
    type=tile_type,
    addrs=0x0000,
    mutex_ids=[0, 1],
)
```

TileGroup可用于单缓冲、双缓冲和N缓冲：

| 配置 | 含义 |
|:---|:---|
| mutex_ids=[0] | 包含一块Tile，无实际轮转效果 |
| mutex_ids=[0, 1] | 两块Tile交替使用，即双缓冲或ping-pong缓冲 |
| mutex_ids=[0, 1, 2, ...] | 三缓冲及更深的N缓冲 |
| mutex_ids=None, depth=N | 创建N块Tile，但不提供自动同步所需的mutex元数据 |

### 配置Tile地址

addrs支持两种形式：

```python
import pypto_pro.language as pl


# 传入单个基地址：各Tile按照单个槽位大小连续排布。
continuous = pl.make_tile_group(
    type=tt,
    addrs=0x0000,
    mutex_ids=[0, 1],
)

# 传入地址列表：显式指定每块Tile的地址。
separate = pl.make_tile_group(
    type=tt,
    addrs=[0x0000, 0x10000],
    mutex_ids=[2, 3],
)
```

传入地址列表时，地址数量必须与TileGroup深度一致。使用单个基地址时，框架根据单块Tile占用的槽位大小计算后续地址。

### 配置mutex和depth

mutex ID是片上缓冲区互斥同步资源的编号，用于协调不同Pipe对Tile的访问。为TileGroup配置mutex_ids，是指定组内每块Tile使用的ID；启用@pypto_pro.language.jit(auto_mutex=True)后，框架根据这些ID自动插入同步操作。也可以只传depth而不传mutex_ids来创建轮转Tile，但此时框架不会分配mutex ID；即使启用auto_mutex=True，也不会为这些Tile自动插入mutex同步，需要开发者自行保证访问时序。

- mutex_ids中的每一项对应一块Tile，ID取值范围为[0, 31]。
- 每块Tile可以绑定一个ID，也可以绑定一个非空ID列表；同一块Tile绑定的多个ID不能重复。
- mutex_ids非空且未指定depth时，TileGroup深度由len(mutex_ids)推导。
- mutex_ids为None或空列表时必须显式指定depth，并由开发者保证访问时序。

分配ID时，在同一个TileGroup内和不同的TileGroup之间，优先为每块Tile分配不同的ID。比如双缓冲使用mutex_ids=[0, 1]，另一块同时参与流水的Tile使用mutex_ids=[2]。这样可避免无关Tile因共用ID而互相等待，更利于发挥流水并行性能。ID不够用时，优先让使用时段不重叠、不会同时参与流水的Tile复用同一ID；尽量不要让可能同时使用的两块Tile共用ID。

如果还显式调用pypto_pro.language.system.mutex_lock/mutex_unlock，应将其ID与自动同步正在使用的ID分开；只有确认两者的使用周期完全不重叠，才能复用同一ID，避免重复获取尚未释放的互斥资源。TileGroup本身不会在运行时主动执行加锁或解锁。完整参数说明请参考[pypto_pro.language.make_tile_group](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)。

## 访问TileGroup中的Tile

TileGroup提供轮转访问和下标访问：

| 访问方式 | 是否推进轮转位置 | 说明 |
|:---|:---:|:---|
| group.next() | 是 | 推进轮转位置并返回下一块Tile；连续调用时循环选择 |
| group.current() | 否 | 返回当前Tile |
| group.previous() | 否 | 返回当前Tile的前一块Tile |
| group[i] | 否 | 返回下标为i的Tile，i可以是编译期整数或运行时整数表达式；运行时下标必须位于[0, depth)范围内 |

TileGroup的初始轮转位置为最后一块Tile，因此第一次调用next()返回第0块Tile；调用next()前，current()返回最后一块Tile。

```python
import pypto_pro.language as pl


double_buffer = pl.make_tile_group(
    type=tt,
    addrs=0x0000,
    mutex_ids=[0, 1],
)

current_tile = double_buffer.current()
next_tile = double_buffer.next()
previous_tile = double_buffer.previous()
first_tile = double_buffer[0]
```

对同一个TileGroup交替调用next()时，需要保证调用顺序与Kernel的流水迭代一致。仅需固定访问某一块Tile时，使用current()或下标访问，避免无意改变轮转位置。。

## 设置Tile的有效形状

当Tensor shape不能被Tile shape整除时，边界Tile只有部分元素有效。TileType省略valid_shape时默认采用动态模式，等同于valid_shape=[-1, -1]，无需为尾块显式填写：

```python
import pypto_pro.language as pl


tail_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)
```

需要表示部分维度为固定有效大小时，可显式指定valid_shape，例如[64, -1]。取得具体Tile后，使用pypto_pro.language.set_validshape设置本次访问的有效范围；每一维必须大于0且不能超过Tile对应维度：

```python
import pypto_pro.language as pl


tile = tile_group.next()
pl.set_validshape(tile, [valid_rows, valid_cols])
```

也可以直接对TileGroup调用set_validshape，批量设置组内所有Tile：

```python
import pypto_pro.language as pl


pl.set_validshape(tile_group, [valid_rows, valid_cols])
```

有效形状通常需要在数据搬入前设置，确保后续搬运使用正确的边界。compact和pad是否需要配置取决于内存空间、layout和具体数据路径。尾块数量、有效形状和多核任务分配参考[尾块处理](tiling/multi_core_tiling.md#尾块处理)。

## 重声明Tile视图

pypto_pro.language.reinterpret可以用新的dtype、shape或layout重新声明已有Tile或TileGroup：

```python
import pypto_pro.language as pl


new_tile = pl.reinterpret(
    tile,
    dtype=pl.DT_INT32,
    shape=[64, 64],
    layout=pl.ND,
)
```

新对象与原Tile共享相同的Buffer地址和大小；该操作不搬运数据，也不进行数值或layout转换。新shape和dtype的存储占用不能超过原Tile绑定的地址范围，调用方还需要确保缓冲区中的物理数据符合新layout。

reinterpret至少需要指定dtype、shape或layout中的一项；显式指定dtype时必须同时指定shape。新视图不会继承原Tile的valid_shape，需要通过pypto_pro.language.set_validshape重新设置运行时有效范围。

对TileGroup执行reinterpret后，新旧TileGroup共享同一套缓冲区轮转状态；任意一方调用next()都会影响另一方。需要独立轮转时，应分别创建TileGroup。详细约束请参考[pypto_pro.language.reinterpret](../../../../api/pro_api/SIMD-API/resource_management/reinterpret.md)。

## 相关内容

Tile上的矢量计算接口请参考[Tile计算](vector_computation/tile_computation.md)，矩阵搬运和计算请参考[Cube计算](cube_computation.md)，跨Pipe手动同步接口请参考[同步API](../../../../api/pro_api/SIMD-API/synchronization/index.md)。
