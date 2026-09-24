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

自动CV并行流水的使用约束分别列在以下各使用步骤后的“说明”中，开发时应结合对应步骤检查。

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

> [!NOTE]说明
>
> - stage函数不能嵌套stage，也不能有返回值；stage调用的普通函数不能返回Tile或Tile Group。
> - stage函数有结构体参数时，需要使用`pypto_pro.language.struct()`声明，不支持`pypto_pro.language.struct_array()`。

### 声明跨核共享Buffer

跨核共享Buffer使用make_tile_group接口进行声明。`mutex_ids`与`fwd_ids`、`bwd_ids`作用于不同范围：`mutex_ids`用于管理核内Tile槽位的流水同步和地址复用，`fwd_ids`、`bwd_ids`用于Cube与Vector之间的数据交接。声明了`fwd_ids`或`bwd_ids`，且该Buffer分别被Cube stage和Vector stage使用时，框架才将其识别为跨核共享Buffer，并插入相应的核间同步。

`mutex_ids`需要为Buffer中的每个Tile配置一个id，因此其数量与Tile数一致。`fwd_ids`和`bwd_ids`的数量不要求始终与Tile数一致：可以为每个Tile分别配置一个id，也可以只配置一个由所有Tile共用的id。共用一个核间同步id会使不同Tile之间的交接串行化，能够减少同步id占用，但可能降低流水并行度。

其中`fwd_ids`为正向同步（生产者写完通知消费者），生产者stage之后插入set、消费者stage之前插入wait；`bwd_ids`为反向同步（消费者用完通知生产者可以覆写），消费者stage之后插入set、生产者stage之前插入wait。只配置`fwd_ids`时，仅保证消费者读到的是生产者写完的数据，不保证生产者下一轮覆写时消费者已经读完。每次迭代使用的Tile按Tile数轮转；核间同步id则按`ids[迭代序号 % len(ids)]`轮转，因此二者的轮转周期可以不同。

`target_memory`仅声明Buffer所在的物理存储空间，不表示是否跨核。跨核共享Buffer支持UB和L1。下面的`mm1_vec_db`是Cube stage将计算结果交给Vector stage的示例；在后文完整示例中，`p_mat_db`由Vector stage写入后交给Cube stage读取。两者的跨核关系都由stage的生产者/消费者关系以及`fwd_ids`、`bwd_ids`确定。

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

> [!NOTE]说明
>
> - 跨核Buffer和核内Buffer的`make_tile_group`声明需要写在kernel函数体内，不能写在stage调用的普通函数中；跨核Buffer仅支持UB和L1 Buffer。
> - 一个跨核Buffer需要恰好被相邻的两个stage使用，两个stage分别位于Cube和Vector上，并构成一对一的生产者和消费者关系。需要把数据送到更后面的stage时，应逐级交接。
> - `fwd_ids`和`bwd_ids`的取值范围为0～15，只能直接写编译期常量整数列表，或使用kernel外绑定到整数列表的变量，不支持切片、拼接、函数调用等表达式。ID数量需要等于Tile数或等于1；等于1时所有Tile共用同一个同步ID，可能降低性能。
> - 复用地址时最多允许两块Buffer复用，双方的Tile数和`mutex_ids`需要一致；`mutex_ids`不同不能形成地址互斥。
> - 跨核Buffer的Tile需要随迭代顺序轮转。按stage在主循环中的书写顺序编号（与第一个stage位于Cube还是Vector无关），写该Buffer的stage编号为奇数时至少需要2个Tile，为偶数时至少需要`preload`个Tile；Tile数不足时会在编译期报错，并提示所需的最小Tile数。
> - 在空间充足时，可以分配多于下限的Tile，并根据实测结果判断是否有性能收益。调整Tile数时，需要同步调整`mutex_ids`、`fwd_ids`和`bwd_ids`的数量以及Buffer占用的地址空间；核间同步ID配置为所有Tile共用的单个ID时除外。

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

> [!NOTE]说明
>
> - 同一条流水的stage调用需要放在同一个for循环内，循环步长必须为正，不支持倒序迭代。每个`section_cube()`或`section_vector()`块只能包含一个stage调用，各stage需要按Cube、Vector严格交替排列；第一个stage与最后一个stage之间不能插入其他语句。
> - 流水循环体内不能直接获取或操作跨核Buffer，以及与跨核Buffer复用地址的核内Buffer，应把这些操作放进stage函数。同一个stage函数在一条流水中只能调用一次，不同stage函数不能重名。
> - 可以用if语句控制stage是否执行，但条件必须是编译期常量。在流水循环外调用`group.next()`取Tile时，需要使用单独的赋值语句，不能嵌入其他表达式；同一个变量只能取一次Tile，需要多个Tile时应使用多个变量。
> - 一个kernel可以包含多个独立的流水循环，但这些循环不能嵌在同一个外层循环中，循环之间需要手动插入全核同步。kernel内不要使用以`_pl_`开头的框架保留变量名。

### 开启流水变换

在pypto_pro.language.jit接口中通过PipelineConfig参数进行配置：


| 参数    | 含义                                                                                                                                                                                                      |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| preload | 表示上游核提前迭代计算的次数，类型为int或list[int]。传入单个整数时，kernel内所有流水循环共用该值；传入列表时，按源码顺序为每个流水循环单独配置。取值为0的循环不做流水改写，只在原串行循环中插入核间同步。 |

**建议流程**：先配置**preload=0**执行，确认串行版本精度正确，再逐步调大preload开启流水。调大preload的同时，需按[声明跨核共享Buffer](#声明跨核共享buffer)后的说明检查各跨核Buffer的Tile数是否够用。多个流水循环时可以只把其中一条配成0（如preload=[0, 2]），单独验证这条流水的串行精度。

调优时应同时观察`preload`和Buffer Tile数对性能的影响。如果增大`preload`或增加Tile后性能没有提升，可将Tile数减少到当前`preload`对应的最小值，以节省片上存储空间；减少后仍需满足[声明跨核共享Buffer](#声明跨核共享buffer)后说明中的Tile数下限，并重新验证精度和性能。

> [!NOTE]说明
>
> - `preload`需要大于等于0；传入列表时，列表长度需要与kernel内流水循环的数量一致。

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

### 完整示例的流水与同步关系

下图对应上述完整示例的`N_ITER=4`和`preload=2`配置。`sN·iM`表示stage N正在处理第M次迭代；两个AIV执行相同的Vector stage，分别处理M方向的一半，因此合并画在一条Vector泳道中。方框宽度和空隙用于示意各stage的耗时与等待关系，不表示真实执行周期；实线表示Forward数据就绪依赖，虚线表示Backward槽位释放依赖。

**图2 完整示例在preload=2时的执行顺序与同步关系**

![自动CV并行流水完整示例的流水与同步关系](../../../figures/pro/pro_auto_parallel_pipeline_example.png "自动CV并行流水完整示例的流水与同步关系")

- **并行preload**：填充阶段先让前级stage处理两个迭代。进入稳态后，Cube侧交错执行`stage1(i+2)`和`stage3(i)`，Vector侧交错执行`stage2(i+2)`和`stage4(i)`，AIC与AIV同时处理不同迭代的数据。主循环结束后，编译器继续排空已经进入流水的`stage3`和`stage4`任务。
- **核间同步**：编译器根据跨核Buffer的`fwd_ids`和`bwd_ids`自动插入`set_cross_core`和`wait_cross_core`。`mm1_vec_db`使用Forward ID 0/1和Backward ID 2/3，`p_mat_db`使用Forward ID 4/5和Backward ID 6/7，`out_vec_db`使用Forward ID 8/9和Backward ID 10/11。Forward事件保证消费者等待数据写完，Backward事件保证生产者等待消费者使用完毕后再覆写对应槽位。
- **核内同步**：`auto_mutex=True`根据各Tile Group的`mutex_ids`管理同一AIC或AIV内部不同流水单元对Tile的读写、槽位轮转和地址复用。`mutex_ids`负责核内同步，`fwd_ids`和`bwd_ids`负责AIC与AIV之间的数据交接，两者的作用范围不同。

本示例的`mm1_vec_db`和`out_vec_db`分别由奇数编号的stage1、stage3写入，因此至少需要2个Tile；`p_mat_db`由偶数编号的stage2写入，Tile数不能小于`preload`。当前`preload=2`，所以三块跨核Buffer均配置为2个Tile。

> [!NOTE]说明
>
> - 本示例按单个1:2 CV执行组设计，因此`block_dim`设置为1。扩展为多个执行组时，需要通过`pypto_pro.language.get_block_idx()`划分各执行组处理的GM数据和输出范围。`block_dim`的完整说明参见[Kernel核函数](../development/kernel_function.md#blockdim的含义与设置)。
