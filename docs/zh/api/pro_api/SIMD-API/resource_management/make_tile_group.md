# pypto_pro.language.make_tile_group

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

一次性创建一组同规格且可轮转复用的Tile，为每块Tile绑定独立地址，并可选建立Tile与一个或多个mutex ID的对应关系。解析器会将TileGroup展开成多个[pypto_pro.language.make_tile](make_tile.md)操作；启用auto_mutex后，框架根据该元数据为涉及这些Tile的后续操作自动插入mutex同步。

缓冲深度由depth或非空mutex_ids的长度确定：

- **1-buffer**（depth为1，或mutex_ids长度为1）：仅包含一块Tile，可使用current()或group[0]访问，不产生实际的轮转效果。
- **2-buffer**（double-buffer/ping-pong）：两块Tile交替使用。
- **3-buffer、4-buffer或N-buffer**：多块Tile循环轮转，适用于通过加深流水隐藏搬运延迟的场景。

与多次调用[pypto_pro.language.make_tile](make_tile.md)相比，make_tile_group统一管理地址排布、轮转位置和mutex元数据，无需手动维护Tile下标与同步关系。TileGroup本身不在运行时执行加锁或解锁；启用auto_mutex后，其mutex映射在编译期用于生成同步操作。

下图以UB中的双缓冲为例，展示单个基地址如何展开成两个连续Tile槽位，以及地址、mutex和轮转访问之间的对应关系。

![make_tile_group双缓冲地址与轮转关系](../../figures/make_tile_group_rotation.jpg "make_tile_group双缓冲地址与轮转关系")

## 函数原型

```python
pypto_pro.language.make_tile_group(
    *,
    type: TileType,
    addrs: Union[int, List[int]],
    mutex_ids: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    depth: Optional[int] = None,
    fwd_ids: Optional[Sequence[int]] = None,
    bwd_ids: Optional[Sequence[int]] = None,
) -> TileGroup
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| type | 输入 | Tile类型描述，[pypto_pro.language.TileType](../basic_data_structures/TileType.md)类型。 |
| addrs | 输入 | Tile地址，int或List[int]类型，必须非负并在编译期确定，且满足对应Buffer的地址对齐要求。传入单个基地址时，第i块Tile的地址为base + i × slot_size，其中slot_size为单块Tile占用的字节数；传入地址列表时，列表长度必须等于mutex_ids的长度或depth的值，并按顺序为每块Tile指定地址。地址列表可用于非连续地址排布。 |
| mutex_ids | 输入 | mutex ID配置，Sequence[int或Sequence[int]]类型，可选，也可传入None或空列表。mutex ID的取值范围为[0, 31]。每块Tile对应的mutex ID数量必须一致，同一块Tile的多个mutex ID不得重复，不同Tile之间可以使用相同的mutex ID。 |
| depth | 输入 | TileGroup深度，int类型，可选，必须为正的编译期整数。mutex_ids为None或空列表时必须指定depth；未指定depth且mutex_ids非空时，由mutex_ids的长度确定；同时指定时，两者必须相等。 |
| fwd_ids | 输入 | 仅用于[自动CV并行流水](../../../../guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.md)，表示核间正向同步的event id，Sequence[int]类型，可选。未开启自动流水时忽略。 |
| bwd_ids | 输入 | 仅用于[自动CV并行流水](../../../../guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.md)，表示核间反向同步的event id，Sequence[int]类型，可选。未开启自动流水时忽略。 |

## 约束说明

- 使用group[i]访问Tile时，i支持编译期整数或运行时整数表达式，例如g[i]、g[(i + 1) % 2]。索引值应位于[0, num_tile)范围内；num_tile为mutex_ids非空时的列表长度，否则为depth。
- 配置非空mutex_ids且启用auto_mutex时，框架根据Tile与mutex的映射插入同步；未启用auto_mutex时，调用方必须自行保证Tile的访问时序。

## 返回值说明

返回TileGroup对象，支持以下访问方式：

| 访问方式 | 说明 |
|---|---|
| group.next() | 推进轮转位置并返回下一块Tile；连续调用时按照TileGroup大小循环选择Tile。 |
| group.current() | 不推进轮转位置，返回当前Tile。 |
| group.previous() | 不推进轮转位置，返回当前Tile的前一块Tile。 |
| group[i] | 返回下标为i的Tile，i支持编译期整数或运行时整数表达式；该操作不读取或改变轮转位置。 |

TileGroup还可直接传给pypto_pro.language.set_validshape，批量设置组内所有Tile的valid_shape。

## 调用示例

### 2-buffer（double-buffer / ping-pong）

```python
import pypto_pro.language as pl

TILE = 128
MM_M, MM_K, MM_N = 256, 128, 256


@pl.jit(auto_mutex=True)
def tile_group_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A Buffer、L0B Buffer和L0C Buffer均使用单Tile组（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1_db.next()      # 双缓冲轮转
                cur_b = b_l1_db.next()
                al = a_left.current()       # 单Tile组
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### 在L1 Buffer中创建4-buffer

```python
import pypto_pro.language as pl

TILE = 128
MM_M, MM_K, MM_N = 256, 128, 256


@pl.jit(auto_mutex=True)
def tile_group_4buf_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1, 2, 3])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5, 6, 7])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[8])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[9])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[10])

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### 使用地址列表创建非连续Buffer

```python
# 两块 Tile 分别落在 0x0 和 0x10000
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
buf = pl.make_tile_group(type=tt, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
```
