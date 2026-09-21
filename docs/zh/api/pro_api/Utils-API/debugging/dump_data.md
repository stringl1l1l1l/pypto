# pypto_pro.language.dump_data

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

调测打印接口，用于打印GM Tensor或Tile的内容，支持全量打印和窗口打印。

- 输入为Tensor（GM全局内存张量）时，打印GM上的Tensor数据
- 输入为Tile（通过make_tile/make_tile_group分配）时，打印Tile数据

Tile通过[pypto_pro.language.make_tile](../../SIMD-API/resource_management/make_tile.md)或[pypto_pro.language.make_tile_group](../../SIMD-API/resource_management/make_tile_group.md)创建。不同内存空间的Tile打印能力不同：

- UB Tile：可直接打印，无需workspace
- L0C Buffer中的Tile：须通过workspace参数中转打印（先将Tile数据写回GM，再打印）
- L1 Buffer、L0A Buffer和L0B Buffer中的Tile：无法直接打印

打印结果直接输出到终端。

## 函数原型

```python
pypto_pro.language.dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[List[int]] = None,
    shapes: Optional[List[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| data | 输入 | 要打印的数据，必须是Tensor（TensorType）或Tile（TileType），其他类型报TypeError。Tensor暂不支持pypto_pro.language.DT_FP4、pypto_pro.language.DT_FP4E2M1和pypto_pro.language.DT_FP4E1M2类型。 |
| offsets | 输入 | 可选，窗口起始偏移（各维）。取值为由整型常量或运行时整型标量表达式组成的序列，长度须等于数据的维数；须与shapes同时提供或同时为None。Tile窗口当前仅支持二维Tile。offsets和shapes均为None时打印全部数据。 |
| shapes | 输入 | 可选，窗口大小（各维）。取值为由整型常量或运行时整型标量表达式组成的序列，长度须等于数据的维数；须与offsets同时提供或同时为None。其中编译期常量必须大于0；Tensor窗口模式还要求最内维stride为编译期常量1。NZ Tensor窗口须保持完整分形：M shape按16对齐，N shape和offset按C0对齐；前导维按batch逐个打印。Tile窗口要求二维且Tile物理shape为编译期常量。 |
| workspace | 输入 | 可选，GM上的临时Tensor，仅用于L0C Buffer中Tile的中转打印。仅当data为L0C Buffer中的Tile时有效；Tensor传入该参数会报ValueError，其他内存空间的Tile会报错。必须是与Tile dtype相同的GM TensorType，容量至少能容纳完整物理Tile。 |
| loc | 输入 | 可选，是否在输出前打印源文件/行号，取值为True或False（默认）。 |
| flag | 输入 | 可选，字符串标签，必须是编译时常量，不能是运行时变量。提供时在dump输出前单独打印一行标记`=== [flag] <flag> ===`，用于区分多个dump点的输出。 |

## 约束说明

### L0C Buffer中的Tile场景

L0C Buffer中的Tile无法像UB Tile那样直接打印。dump_data会先将Tile数据写回GM上的workspace Tensor，再打印该Tensor，并须满足以下约束：

- workspace必须是TensorType，可以是核函数参数中的pypto_pro.language.Tensor，也可以通过pypto_pro.language.make_tensor从pypto_pro.language.Ptr构造，不能是Tile
- workspace的dtype必须与待dump的Tile dtype一致
- 无论全量dump还是窗口dump，都会先将L0C Buffer中的完整Tile写入workspace，因此其容量须至少覆盖完整物理Tile
- L0C Buffer中的Tile必须为二维，且物理shape必须是编译期常量

## 返回值说明

无。

## 调用示例

### Tensor输入

```python
import pypto_pro.language as pl


@pl.jit()
def dump_data_tensor_full_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i * 10
        pl.dump_data(out)
```

全量dump输出示例：

```text
=== [dump_tensor] dtype: int32, Layout: ND, shape=[1,1,1,1,16] ===
  Batch [0, 0, 0]:
0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
```

动态偏移示例：

```python
# 循环变量作为偏移（动态）
for i in pl.range(0, 16, 4):
    pl.dump_data(out, offsets=[i], shapes=[4])

# get_block_idx() 作为偏移（动态）
vidx = pl.get_block_idx()
pl.dump_data(out, offsets=[vidx * 4], shapes=[4])
```

带标签输出（用于区分多个dump点）：

```python
pl.dump_data(out, flag="checkpoint_A")
```

标签输出示例：

```text
=== [flag] checkpoint_A ===
=== [dump_tensor] dtype: int32, Layout: ND, shape=[1,1,1,1,16] ===
  Batch [0, 0, 0]:
0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
```

### UB Tile输入

UB Tile可直接打印，无需workspace。L1 Buffer、L0A Buffer和L0B Buffer中的Tile无法直接打印。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[8, 8], pl.DT_INT32]
    b: pl.Tensor[[8, 8], pl.DT_INT32]
    out: pl.Tensor[[8, 8], pl.DT_INT32]
):
    tt = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        pl.store(out, tc, [0, 0])
```

全量dump输出示例：

```text
=== [dump_tile] dtype: int32, shape=[8,8], valid=[8,8], Layout: ND ===
0 2 4 6 8 10 12 14
16 18 20 22 24 26 28 30
32 34 36 38 40 42 44 46
48 50 52 54 56 58 60 62
64 66 68 70 72 74 76 78
80 82 84 86 88 90 92 94
96 98 100 102 104 106 108 110
112 114 116 118 120 122 124 126
```

### L0C Buffer中的Tile输入（需要workspace）

L0C Buffer中的Tile需要通过workspace参数提供GM上的临时Tensor进行中转。

全量dump：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_fp16_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)

        # 打印整个 Acc Tile（需要 workspace 中转）
        pl.dump_data(ac, workspace=workspace)

        # 打印窗口（带 offsets/shapes 和 workspace）
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=workspace)

        pl.store(out, ac, [0, 0])
```

窗口模式输出示例（offsets=[16, 16], shapes=[8, 8]）：

```text
=== [TPRINT Acc Tile Window] Data Type: float32, Layout: NZ, TileType: Acc ===
  Source Shape: [64, 64], Window Offsets: [16, 16], Requested Shape: [8, 8], Valid Shape: [8, 8]
   2.435661  21.525450  -2.534927  -3.354072  -1.637453  -7.538389  -5.316622   7.327333
   3.319994   4.183826   9.192725 -15.309023   5.075872  15.763545  -1.755892  -7.553324
  -6.676492  -1.058733  -2.251584  -8.538083  -0.172604   9.005786  -1.326701   7.341187
   5.795816 -12.892869   3.342661   3.139680  10.270340  -0.026452  -2.230551   3.213134
   3.064780  -5.402464  -0.289040  -4.588926  -0.931392 -12.228477 -20.040319  10.303446
  -5.076264   0.564521  11.335535  -0.019537  -1.963741   4.344845  -0.789701   7.402071
 -13.048984  -7.837986 -16.793615   5.720566  -6.111812 -27.283802   1.088718  -7.852593
 -10.569035   7.459199   5.887267   7.939989   1.122919   4.743242 -10.458792  -0.729014
```

在分块循环中使用窗口dump：

```python
for i in pl.range(0, 256, 64):
    for j in pl.range(0, 256, 64):
        # ... matmul ...
        pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32], workspace=workspace)
        pl.store(out, ac, [i, j])
```
