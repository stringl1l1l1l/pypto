# 自动CV并行流水

## 功能说明

CV融合算子中，cube和vector的计算相互依赖，若按串行流水执行，一个核工作时另一个核只能空等。为了提升CV之间的并行度，让上游核提前若干次迭代计算，下游核则处理上游已经算好的较早迭代的数据，两个核错开节奏、同时工作，从而把彼此的等待时间掩盖掉。

自动CV并行流水是pypto_pro.language.jit的一项编译期变换，包含两个能力：

- **自动流水排布**：把用户手写的CV融合算子串行流水kernel代码，自动改写成并行流水版本，提升性能；
- **自动核间同步**：用户开发的串行版本不需要手写任何CV核间同步指令，并行流水版本会插入全部所需的核间同步。

下图对比了同一个算子（4个stage，cube/vector交替）在串行流水与并行流水（cube提前2次执行）下的执行图：

**图1 串行流水与并行流水的执行节奏对比**

![串行流水与并行流水的执行节奏对比](../../../figures/pro/pro_parallel_pipeline_serial_vs_parallel.png "串行流水与并行流水的执行节奏对比")

图中`sN·iM`表示第N个stage正在处理第M次迭代的数据，格子宽度代表该stage的耗时（各stage耗时不同，图中为示意值）。

需要注意的是，流水化的收益上限由**较慢的那个核**决定：图中vector侧单次迭代的耗时高于cube侧，稳态节奏就由vector侧决定，cube侧会出现等待间隙。stage划分越均衡（两个核的耗时越接近），流水填充得越满，收益越高。此外流水的建立和排空各需要若干拍，迭代次数越多，这部分开销占比越低。

## 使用方法

### 编写Stage函数

需要用户将算子划分为若干个计算流程，每个计算流程对应一个stage函数，通过@pypto_pro.language.pipeline.stage装饰器进行标识。

```python
import pypto_pro.language as pl

@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube：mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    ...  # 普通的 Tile/Buffer 操作，不用写任何同步
```

### 声明跨核共享Buffer

跨核共享Buffer使用make_tile_group接口进行声明，通过fwd_ids和bwd_ids参数配置核间正反向同步id，若未配置，则不会插入对应的核间同步。

其中fwd_ids为正向同步（生产者写完通知消费者），生产者stage之后插入set、消费者stage之前插入wait；bwd_ids为反向同步（消费者用完通知生产者可以覆写），消费者stage之后插入set、生产者stage之前插入wait。只配置fwd_ids时，仅保证消费者读到的是生产者写完的数据，不保证生产者下一轮覆写时消费者已经读完。每次迭代实际使用的id按`ids[迭代序号 % len(ids)]`轮转取用。

```python
import pypto_pro.language as pl

mm1_vec_db = pl.make_tile_group(
    type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    addrs=0x0000,
    mutex_ids=[12, 13],
    fwd_ids=[0, 1],
    bwd_ids=[2, 3],
)
```

Tile数不能随意取，与上游核提前迭代计算的次数相关，且直接影响流水性能，具体规则参见[使用约束](#使用约束)。

### 编写主循环

主循环内stage函数按照依赖关系顺序进行书写，需要保证stage之间为CV交替。

```python
import pypto_pro.language as pl

for ki in pl.range(0, N_ITER):
    with pl.section_cube():
        stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
    with pl.section_vector():
        stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
    with pl.section_cube():
        stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
    with pl.section_vector():
        stage4(ki, sub_id, out, out_vec_db)
```

一个kernel内可以写多个独立的for循环，每个循环各自做流水变换，互不影响，多个循环按源码顺序先后执行，preload可以逐循环配置（见下一节）。

```python
import pypto_pro.language as pl

for ki in pl.range(0, N_ITER):        # 第一条流水
    with pl.section_cube():
        stage1(ki, ...)
    with pl.section_vector():
        stage2(ki, ...)

pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

for kj in pl.range(0, M_ITER):        # 第二条流水
    with pl.section_cube():
        stage3(kj, ...)
    with pl.section_vector():
        stage4(kj, ...)
```

### 开启流水变换

在pypto_pro.language.jit接口中通过PipelineConfig参数进行配置：


| 参数    | 含义                                                                                                                                                                                                      |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| preload | 表示上游核提前迭代计算的次数，类型为int或list[int]。传入单个整数时，kernel内所有流水循环共用该值；传入列表时，按源码顺序为每个流水循环单独配置。取值为0的循环不做流水改写，只在原串行循环中插入核间同步。 |

**建议流程**：先配置**preload=0**执行，确认串行版本精度正确，再逐步调大preload开启流水。调大preload的同时，需按[使用约束](#使用约束)检查各跨核Buffer的Tile数是否够用。多个流水循环时可以只把其中一条配成0（如preload=[0, 2]），单独验证这条流水的串行精度。

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=0))
def pipeline_demo_kernel(...):
    ...
```

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(...):
    ...
```

```python
import pypto_pro.language as pl

# kernel内有两个流水循环，第一条preload=1，第二条preload=2
@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[1, 2]))
def pipeline_demo_kernel(...):
    ...
```

### 查看生成的代码

框架自动生成的并行流水代码保存在编译产物目录下，文件名为pipeline_generated.py，用户可以在该代码的基础上继续修改调试。

## 使用约束

- 同一条流水的stage调用需要放在同一个for循环内。一个kernel可以有多个独立的流水循环，循环之间需要用户手动插入全核同步，两条流水循环不能嵌在同一个外层循环里。
- 流水循环的步长必须为正，不支持倒序迭代。
- preload取值必须大于等于0；传入列表时，列表长度必须与kernel内流水循环的个数相同。
- 不支持stage嵌套stage。
- 跨核Buffer和核内Buffer的make_tile_group声明必须写在kernel函数体内，不支持在被stage调用的普通函数里声明。
- stage函数不支持有返回值。
- stage调用的普通函数不能返回tile或tile group。
- 每个with pypto_pro.language.section_cube()/pypto_pro.language.section_vector()块里只放单个stage调用，且stage调用需要严格按cube/vector交替排列（C→V→C→V…），不允许连续两个stage落在同一个核上。
- 流水循环体内，第一个stage与最后一个stage之间不允许插入其他语句。
- 流水循环（含stage调用的那个for循环）体内不允许获取或操作跨核Buffer，以及与跨核Buffer地址复用的核内Buffer，请把它们放进stage函数体内。
- 在流水循环之外取Tile时，必须单独写成一条赋值语句（slot = group.next()），不能嵌在更大的表达式里；同一个变量只能取一次Tile，需要多个Tile请用多个变量。
- 同一个stage函数在一条流水循环内只能调用一次，不允许重名stage。
- 允许通过if语句判断stage执行场景，但分支条件必须为编译期常量。
- fwd_ids/bwd_ids取值范围为0~15。
- fwd_ids/bwd_ids只支持两种写法：直接写整数列表（fwd_ids=[0, 1]），或写一个在kernel外绑定到整数列表的变量名（IDS = [0, 1] … fwd_ids=IDS）。元素必须是编译期常量整数，不支持切片、拼接、函数调用等表达式形式。
- fwd_ids/bwd_ids的长度只能等于该Buffer的Tile数，或者等于1。等于1时多个Tile共用同一个同步id，交接会被串行化（性能下降但结果正确），用于同步id不够分配的场景。
- 允许跨核Buffer之间、跨核Buffer与核内Buffer之间进行地址复用，但最多允许两块Buffer复用，且复用双方的Tile数需要一致。
- 地址复用的Buffer之间必须声明相同的mutex_ids。mutex锁的是地址，不同的id等于没有互斥。
- 一个跨核Buffer（通过fwd_ids/bwd_ids标记）需要恰好被两个stage使用，且这两个stage分别在cube和vector上，构成一对一的生产者/消费者关系。
- 跨核Buffer的生产者stage与消费者stage必须是主循环中相邻的两个stage，即消费者紧跟在生产者之后，不允许跨过中间的stage直接交接（如stage1写、stage4读）。需要把数据送到更后面的stage时，请逐级交接。
- 跨核Buffer的Tile数由preload决定，不能随意取：按stage在主循环中的书写顺序编号（1、2、3……，与第一个stage在cube还是vector上无关），写这块Buffer的stage编号为奇数时至少需要2个Tile，为偶数时至少需要preload个Tile。以本文的[调用示例](#调用示例)（stage1 Cube → stage2 Vector → stage3 Cube → stage4 Vector）为例：mm1_vec_db（stage1写）与out_vec_db（stage3写）始终为2个Tile，与preload无关；p_mat_db（stage2写）至少需要preload个Tile，即preload=3时至少3个、preload=4时至少4个。Tile数不足时编译期报错，并给出该Buffer所需的最小Tile数。
- 上述规则给出的是Tile数的下限。在Buffer空间充足时，可以分配多于该下限的Tile，用于进一步提升性能。增加Tile数时，mutex_ids的长度、fwd_ids/bwd_ids的长度（配成单个共用id的情况除外）以及Buffer占用的地址空间都需要同步调整。
- 跨核Buffer的Tiles必须随迭代顺序轮转。
- stage函数如果有结构体参数，请使用pypto_pro.language.struct()声明，不支持pypto_pro.language.struct_array()。
- 跨核Buffer仅支持UB和L1 Buffer。
- 以_pl_开头的变量名为框架保留，kernel内不要使用。

## 调用示例

```python
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE = 64
TILE_HALF = TILE // 2  # dual mode 沿 M 劈开，每个 AIV 处理一半
N_ITER = 4


@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube：mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    b_slot = b_l1.current()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, cur_a)
    pl.move(right, b_slot)
    pl.matmul(acc, left, right)
    mm1_vec = mm1_vec_db.next()
    # DualModeSplitM：[TILE, TILE] 的 acc 沿 M 劈开，每个 AIV 拿 [TILE_HALF, TILE]
    pl.move(mm1_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db):
    """Vector：对本 AIV 那半做 relu，转 NZ 后按行偏移 insert 回 L1 Buffer"""
    mm1_vec = mm1_vec_db.next()
    relu_vec = relu_vec_db.next()
    pl.relu(relu_vec, mm1_vec)
    relu_nz = relu_nz_db.next()
    pl.move(relu_nz, relu_vec)  # ND -> NZ，insert要求源Tile为NZ
    p_mat = p_mat_db.next()
    pl.insert(p_mat, relu_nz, [sub_id * TILE_HALF, 0])


@pl.pipeline.stage
def stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db):
    """Cube：mm2 = relu(mm1) @ D"""
    d_slot = d_l1.current()
    p_mat = p_mat_db.next()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, p_mat)
    pl.move(right, d_slot)
    pl.matmul(acc, left, right)
    out_vec = out_vec_db.next()
    pl.move(out_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage4(ki, sub_id, out, out_vec_db):
    """Vector：每个 AIV 写回本迭代结果的一半"""
    out_vec = out_vec_db.next()
    pl.store(out, out_vec, [ki * TILE + sub_id * TILE_HALF, 0])


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(
    a: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    d: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    out: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
):
    # ===== 跨核共享 Buffer：声明 fwd_ids/bwd_ids，框架据此自动插同步 =====
    mm1_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[12, 13], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x8000, mutex_ids=[14, 15], fwd_ids=[4, 5], bwd_ids=[6, 7],
    )
    out_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[16, 17], fwd_ids=[8, 9], bwd_ids=[10, 11],
    )

    # ===== Cube 侧局部 Buffer =====
    with pl.section_cube():
        a_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=[2],
        )
        d_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x14000, mutex_ids=[3],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[4, 5],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0000, mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
            ),
            addrs=0x0000, mutex_ids=[8, 9, 10, 11],
        )
        # B、D 在循环外一次搬入，全程复用
        b_slot = b_l1.current()
        d_slot = d_l1.current()
        pl.load(b_slot, b, [0, 0])
        pl.load(d_slot, d, [0, 0])

    # ===== Vector 侧局部 Buffer =====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        relu_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x8000, mutex_ids=[18, 19],
        )
        relu_nz_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_HALF, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
            ),
            addrs=0x18000, mutex_ids=[20, 21],
        )

    # ===== 流水循环：按 cube/vector 交替调用 4 个 stage =====
    for ki in pl.range(0, N_ITER):
        with pl.section_cube():
            stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
        with pl.section_vector():
            stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
        with pl.section_cube():
            stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
        with pl.section_vector():
            stage4(ki, sub_id, out, out_vec_db)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pipeline_demo_kernel():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    a = torch.randn(N_ITER * TILE, TILE, device=device, dtype=torch.float32)
    b = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    d = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    out = torch.zeros(N_ITER * TILE, TILE, device=device, dtype=torch.float32)

    pipeline_demo_kernel[None, 1](a, b, d, out)
    torch.npu.synchronize()

    ref = torch.relu(a @ b) @ d
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```

> [!NOTE]说明
>
> - 本示例按单个1:2 CV执行组设计，因此`block_dim`设置为1。扩展为多个执行组时，需要通过`pypto_pro.language.get_block_idx()`划分各执行组处理的GM数据和输出范围。`block_dim`的完整说明参见[Kernel核函数](../development/kernel_function.md#blockdim的含义与设置)。
