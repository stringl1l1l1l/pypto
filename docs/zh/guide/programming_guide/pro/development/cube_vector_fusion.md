# CV融合计算

PyPTO Pro支持在同一个Kernel中组合Cube计算与Vector计算。本文介绍混合Kernel的执行模型、跨核数据传递和同步方式，并依次提供基础的Matmul+Softmax示例和使用VF、预取流水的FlashAttention进阶示例。

## 执行模型

Kernel同时包含[`pypto_pro.language.section_cube()`](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md)和[`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md)时，PyPTO Pro将其作为混合Kernel执行。一个逻辑Block对应一个AIC和两个AIV，AIC执行Cube域，两个AIV执行相同的Vector域代码，并通过[`pypto_pro.language.get_subblock_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_idx.md)区分各自负责的数据。

| 执行单元 | 主要计算 | 片上存储 | 任务索引 |
|:---|:---|:---|:---|
| AIC | 矩阵乘加 | L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer | [`pypto_pro.language.get_block_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_idx.md) |
| AIV0/AIV1 | 逐元素、归约、数据重排及Reg计算 | UB、Vector Register | `pypto_pro.language.get_block_idx()`与`pypto_pro.language.get_subblock_idx()` |

混合Kernel中，`block_dim`用于配置AIC/AIV执行组数；AIC数量为[`pypto_pro.language.get_block_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md)返回的实际值，AIV数量还需要乘以[`pypto_pro.language.get_subblock_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_num.md)。多执行组场景下，Cube侧和Vector侧必须使用一致的任务映射，避免不同执行组读写同一中间结果。`block_dim`的详细含义参见[Kernel核函数](kernel_function.md#blockdim的含义与设置)，执行组映射参见[SIMD编程范式](../programming_paradigm/SIMD/programming_paradigm.md#纯vector纯cube与混合kernel)，核间切分参见[多核Tiling切分](tiling/multi_core_tiling.md)。

## 跨核中间数据传递

AIC和AIV可以通过片上Buffer或GM中的workspace传递中间数据。两种方式的数据路径和同步要求不同，应根据中间数据的shape、dtype、layout、容量和使用方式选择。

### 通过片上Buffer传递

片上通路不需要将中间结果写回GM，支持以下传递方向：

- Cube → Vector：通过[`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md)将L0C Buffer中的结果搬到UB。一个AIC向两个AIV分发数据时，可以使用[`pypto_pro.language.AccToVecMode`](../../../../api/pro_api/SIMD-API/basic_data_structures/AccToVecMode.md)按M轴或N轴拆分。
- Vector → Cube：先在UB中得到Cube需要的ND/NZ布局，再通过[`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md)或[`pypto_pro.language.insert`](../../../../api/pro_api/SIMD-API/memory_data_movement/insert.md)写入L1 Buffer。

生产端和消费端需要使用一致的中间Tile shape、dtype和layout。一个AIC向两个AIV分发数据时，还需要确定沿M轴或N轴拆分，并按相同顺序使用轮转Buffer。

### 通过GM传递

中间结果也可以先写入GM中的workspace，再由消费端搬入：

```text
Cube：L0C Buffer ──► workspace（GM）──► UB：Vector
Vector：UB  ──► workspace（GM）──► L1 Buffer：Cube
```

workspace可以保存完整或分块的中间Tensor，允许生产端和消费端使用不同的分块顺序，也支持重复读取，但会增加GM读写。并发的逻辑Block、流水迭代或AIV子块应使用互不重叠的区域；若复用同一区域，生产端必须等上一轮读取结束后再覆写。

## 建立跨核同步

同步信号只表达执行依赖，不负责搬运数据。无论中间结果位于片上Buffer还是workspace，只要一个核生产、另一个核消费，就必须保证消费者在数据就绪后读取；Buffer将被循环复用时，还必须保证生产者在消费者使用完毕后才能覆写。`pypto_pro.language.section_cube()`与`pypto_pro.language.section_vector()`在源码中的先后顺序本身不能替代核间同步，[`pypto_pro.language.jit(auto_mutex=True)`](compilation_and_execution/JIT_compilation.md#jit装饰器参数)也只负责Tile相关的核内流水同步；未启用自动CV流水时，核间依赖仍需显式描述。

### 显式同步

[`pypto_pro.language.system.set_cross_core`](../../../../api/pro_api/SIMD-API/synchronization/set_cross_core.md)在生产流水的前序操作完成后发送事件，[`pypto_pro.language.system.wait_cross_core`](../../../../api/pro_api/SIMD-API/synchronization/wait_cross_core.md)阻塞消费流水的后续操作。`pipe`分别表示发送前需要完成的流水和等待时需要阻塞的流水，两端不要求相同。

默认的[`pypto_pro.language.CrossCoreSyncMode.INTRA_BLOCK`](../../../../api/pro_api/SIMD-API/basic_data_structures/CrossCoreSyncMode.md)模式用于同一个逻辑Block内的AIC/AIV同步：

- AIC → AIV：AIC发送数据就绪事件，AIV0和AIV1分别等待。
- AIV → AIC：AIV0和AIV1分别发送完成事件，AIC等待两路事件。

需要只与一个AIV配对时可使用`pypto_pro.language.CrossCoreSyncMode.UNICAST_BLOCK`；`pypto_pro.language.CrossCoreSyncMode.INTER_SUBBLOCK`只同步同一逻辑Block内的AIV0/AIV1；`pypto_pro.language.CrossCoreSyncMode.INTER_BLOCK`同步多个逻辑Block中的同类核，不会在AIC与AIV之间建立依赖。同步模式必须按实际生产者和消费者选择，不能仅根据“是否跨核”判断。

例如，Cube写完workspace后，从FIX流水发送事件；Vector在MTE2流水读取workspace前等待该事件：

```python
import pypto_pro.language as pl


with pl.section_cube():
    pl.store(workspace, cube_result, offsets)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

with pl.section_vector():
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
    pl.load(vector_input, workspace, offsets)
```

事件必须成对出现，并保证所有参与核在所有运行时分支中都能到达同步点，否则可能死锁。同一个事件ID再次用于另一组依赖前，前一组SET/WAIT必须全部消费完成。循环复用同一Buffer时，只设置“数据已就绪”的正向同步并不充分，还需要“数据已消费”的反向同步，或者为相邻迭代使用不同Buffer槽位。

### 自动同步与并行流水

自动同步与并行流水的配置和使用方法参见[自动CV并行流水](../advanced_programming/auto_parallel_pipeline.md)。使用自动流水需要满足该文档中的使用约束；不满足时，应手动编排流水，并使用上述`set_cross_core`和`wait_cross_core`显式配置核间同步。

## 基础示例：Matmul+Softmax

下面实现$\operatorname{Softmax}(A \times B)$。矩阵乘法$S=A\times B$在Cube侧执行，按行Softmax在Vector侧执行：

$$
out_{i,j} = \frac{e^{S_{i,j} - m_i}}{\sum_{j} e^{S_{i,j} - m_i}}, \qquad m_i = \max_{j} S_{i,j}
$$

该实现将$S$保存在workspace中。各逻辑Block沿M轴分配互不重叠的行块：Cube先写完本组负责的$S$，对应的两个AIV再读取这些行并完成Softmax。示例直接调用Kernel，使用当前Stream默认的可用执行组数。

- $A$的shape为$[M,128]$，$B$的shape为$[128,N]$，M/N为运行时动态维度；K固定为128。逻辑Block沿M轴以64行为单位按`get_block_num()`返回的实际执行组数跨步分配行块。
- Cube输出Tile的shape为$[64,64]$，因此每个M行块沿N轴每64列计算一次，并将结果写入shape为$[M,N]$的workspace对应区域。
- Softmax需要沿完整N轴获得每行最大值和指数和。Vector无法一次把完整N轴放入UB，因此按64列遍历三次：第一次合并行最大值，第二次累加指数和，第三次归一化并写回。
- 一个逻辑Block包含两个AIV。两个AIV沿本组的M行块切分，各处理最多32行；M/N尾块通过`valid_shape`传递到Cube和Vector两侧。
- 每组的Cube写完其负责的workspace区域后发送一次事件，同组Vector读取前等待该事件。Cube与Vector在同组内不按N块重叠执行。

### Tile API实现

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128

@pl.jit(auto_mutex=True)
def matmul_softmax_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                          b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                          workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    M_TILES = (M + TILE_M - 1) // TILE_M
    N_TILES = (N + TILE_N - 1) // TILE_N

    # ---- Cube Tile：计算 S = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：每个 AIV 处理最多 32 个 M 行，沿 N 块分三遍完成 Softmax ----
    tt_vec = pl.TileType(shape=[VEC_ROWS, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_red = pl.TileType(shape=[VEC_ROWS, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    # 行归约使用 [M半块, 1] 的 DN 视图，逐元素合并使用同一内存的行主序视图。
    tt_red_rm = pl.TileType(shape=[1, VEC_ROWS], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec)

    qk_vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[5])
    tmp_vec = pl.make_tile_group(type=tt_vec, addrs=0x2000, mutex_ids=[6])
    exp_vec = pl.make_tile_group(type=tt_vec, addrs=0x4000, mutex_ids=[7])
    red_vec = pl.make_tile_group(type=tt_red, addrs=0x6000, mutex_ids=[8])
    red_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6000, mutex_ids=[9])
    global_max = pl.make_tile_group(type=tt_red, addrs=0x6100, mutex_ids=[10])
    global_max_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6100, mutex_ids=[11])
    global_sum = pl.make_tile_group(type=tt_red, addrs=0x6200, mutex_ids=[12])
    global_sum_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6200, mutex_ids=[13])

    # ==== Cube：M轴分核，N轴分块，结果写入workspace ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_base = mi * TILE_M
            valid_m = pl.min(TILE_M, M - m_base)
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
                pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
                pl.load(cur_a_l1, a, [m_base, 0])
                pl.load(cur_b_l1, b, [0, n_off])
                pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
                pl.move(cur_a_l0a, cur_a_l1)
                pl.move(cur_b_l0b, cur_b_l1)
                pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
                pl.store(workspace, cur_qk_l0c, [m_base, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector：在S上沿N方向分块计算Softmax ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_off = mi * TILE_M + sub_id * VEC_ROWS
            if m_off < M:
                valid_rows = pl.min(VEC_ROWS, M - m_off)
                cur_qk = qk_vec.current()
                cur_tmp = tmp_vec.current()
                cur_exp = exp_vec.current()
                cur_red = red_vec.current()
                cur_red_rm = red_rm.current()
                cur_global_max = global_max.current()
                cur_global_max_rm = global_max_rm.current()
                cur_global_sum = global_sum.current()
                cur_global_sum_rm = global_sum_rm.current()
                pl.set_validshape(cur_red, [valid_rows, 1])
                pl.set_validshape(cur_red_rm, [1, valid_rows])
                pl.set_validshape(cur_global_max, [valid_rows, 1])
                pl.set_validshape(cur_global_max_rm, [1, valid_rows])
                pl.set_validshape(cur_global_sum, [valid_rows, 1])
                pl.set_validshape(cur_global_sum_rm, [1, valid_rows])

                # 第一遍：合并所有 N 块的行最大值。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.maximum(cur_red, cur_qk, cur_tmp, dim=0)
                    if nj == 0:
                        pl.mul(cur_global_max_rm, cur_red_rm, 1.0)
                    else:
                        pl.maximum(cur_global_max_rm, cur_global_max_rm, cur_red_rm)

                # 第二遍：基于全局最大值累加所有 N 块的 exp 和。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                    pl.set_validshape(cur_exp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                    pl.exp(cur_exp, cur_exp)
                    pl.sum(cur_red, cur_exp, cur_tmp, dim=0)
                    if nj == 0:
                        pl.mul(cur_global_sum_rm, cur_red_rm, 1.0)
                    else:
                        pl.add(cur_global_sum_rm, cur_global_sum_rm, cur_red_rm)

                # 第三遍：逐块归一化并直接写回 out。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_exp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                    pl.exp(cur_exp, cur_exp)
                    pl.expand_div(cur_exp, cur_exp, cur_global_sum, dim=0)
                    pl.store(out, cur_exp, [m_off, n_off])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 144, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax kernel passed!")
```

#### 实现说明

- L0A Buffer、L0B Buffer和L0C Buffer中的Tile设置`compact=1`，使M/N尾块在L1 Buffer→L0 Buffer和L0C Buffer→GM路径上按当前有效尺寸使用紧凑步长；两侧通过`pypto_pro.language.set_validshape`传递实际M/N范围。
- Vector归约产生`[M半块,1]`的DN结果，用于`pypto_pro.language.expand_sub`和`pypto_pro.language.expand_div`按行广播；同一UB地址同时建立`[1,M半块]`行主序视图，用于跨N块合并最大值和指数和。
- `pypto_pro.language.jit(auto_mutex=True)`管理各Tile的核内mutex，但Cube与Vector之间的数据就绪关系仍由`pypto_pro.language.system.set_cross_core`和`pypto_pro.language.system.wait_cross_core`建立。
- Host测试将随机输入缩小到0.1倍，以避免Softmax高度饱和放大浮点舍入差异。该缩放仅用于测试，不属于Kernel计算。

SIMD编程模型参见[SIMD编程范式](../programming_paradigm/SIMD/programming_paradigm.md)。

### VF API实现

VF（Vector Function，矢量函数）使用[`@pypto_pro.language.vector_function`](vector_computation/reg_computation.md#vf函数与执行域)定义，并通过[`vf.*`接口](../../../../api/pro_api/SIMD-API/reg_computation/index.md)在UB与矢量寄存器之间读写数据。下面的版本沿用前面的Cube计算、workspace布局和核间同步，Vector侧改用VF计算Softmax。`[M半块,N块]`先在UB中转置为`[N块,64]`，使每个N位置对应一个FP32矢量寄存器，寄存器中的lane对应不同的M行。

VF相关的基础概念和接口说明参见[Reg计算](vector_computation/reg_computation.md)。

Vector侧的数据组织如下：

- `qk_nd`从workspace载入`[M半块,N块]`，再转置为`qk_dn[N块,64]`，使每个N位置对应一个64-lane FP32寄存器。
- `global_max`和`global_sum`在UB中保存跨N块的逐行状态，三遍计算分别更新最大值、指数和与归一化结果。
- M尾块通过`vf.update_mask(valid_rows)`限制有效lane，N尾块通过VF循环上界`valid_n`限制实际处理位置。
- VF写入的状态被后续VF或Vector操作读取前，使用[`vf.mem_bar`](../../../../api/pro_api/SIMD-API/reg_computation/data_movement/mem_bar.md)并设置`mode=pypto_pro.language.MemBarMode.VST_VLD`保证局部存储顺序；MTE与Vector之间的Tile依赖仍由mutex管理。

```python
import os
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128
VF_LANES = 64
NEG_INF = -1e30

@pl.vector_function
def softmax_vf_init(global_max, global_sum, valid_rows: pl.DT_INT64):
    """初始化跨 N 块保存的逐行最大值与指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    max_reg = vf.full(NEG_INF, preg, dtype=pl.DT_FP32)
    sum_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(global_max, max_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_update_max(src_dn, global_max,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """把当前 N 块合并到逐行全局最大值。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        max_reg = vf.max(max_reg, src_reg, preg)
    vf.store_align(global_max, max_reg, preg)


@pl.vector_function
def softmax_vf_update_sum(src_dn, global_max, global_sum,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """基于全局最大值累加当前 N 块的指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        sum_reg = vf.add(sum_reg, exp_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_normalize(src_dn, dst_dn, global_max, global_sum,
                         valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """使用完整 N 轴的最大值与指数和归一化当前 N 块。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        out_reg = vf.div(exp_reg, sum_reg, preg)
        vf.store_align(dst_dn + ni * VF_LANES, out_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def matmul_softmax_vf_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                             b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                             out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                             workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    M_TILES = (M + TILE_M - 1) // TILE_M
    N_TILES = (N + TILE_N - 1) // TILE_N

    # ---- Cube Tile：计算 S = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：TTRANS 前后都使用 64x64 物理 Tile，VF 用 mask 收窄 M 尾块 ----
    tt_vf_block = pl.TileType(shape=[VF_LANES, TILE_N], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec)
    tt_vf_state = pl.TileType(shape=[1, VF_LANES], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec)

    qk_nd = pl.make_tile_group(type=tt_vf_block, addrs=0x0000, mutex_ids=[5])
    qk_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x4000, mutex_ids=[6])
    out_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x8000, mutex_ids=[7])
    out_nd = pl.make_tile_group(type=tt_vf_block, addrs=0xC000, mutex_ids=[8])
    global_max = pl.make_tile_group(type=tt_vf_state, addrs=0x10000, mutex_ids=[9])
    global_sum = pl.make_tile_group(type=tt_vf_state, addrs=0x10100, mutex_ids=[10])

    # ==== Cube：M轴分核，N轴分块，结果写入workspace ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_base = mi * TILE_M
            valid_m = pl.min(TILE_M, M - m_base)
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
                pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
                pl.load(cur_a_l1, a, [m_base, 0])
                pl.load(cur_b_l1, b, [0, n_off])
                pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
                pl.move(cur_a_l0a, cur_a_l1)
                pl.move(cur_b_l0b, cur_b_l1)
                pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
                pl.store(workspace, cur_qk_l0c, [m_base, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: 在转置后的 [N块,64] UB Tile 上调用 VF ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_off = mi * TILE_M + sub_id * VEC_ROWS
            if m_off < M:
                valid_rows = pl.min(VEC_ROWS, M - m_off)
                cur_qk_nd = qk_nd.current()
                cur_qk_dn = qk_dn.current()
                cur_out_dn = out_dn.current()
                cur_out_nd = out_nd.current()
                cur_global_max = global_max.current()
                cur_global_sum = global_sum.current()
                pl.set_validshape(cur_global_max, [1, valid_rows])
                pl.set_validshape(cur_global_sum, [1, valid_rows])
                softmax_vf_init(cur_global_max, cur_global_sum, valid_rows)

                # 第一遍：把各 N 块转为 [N块,64]，逐 lane 更新每行最大值。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_update_max(cur_qk_dn, cur_global_max, valid_rows, valid_n)

                # 第二遍：基于全局最大值逐 lane 累加完整 N 轴的指数和。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_update_sum(cur_qk_dn, cur_global_max, cur_global_sum,
                                          valid_rows, valid_n)

                # 第三遍：VF 归一化后转回 [M半块,N块] 并写回 GM。
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.set_validshape(cur_out_dn, [valid_n, valid_rows])
                    pl.set_validshape(cur_out_nd, [valid_rows, valid_n])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_normalize(cur_qk_dn, cur_out_dn, cur_global_max,
                                         cur_global_sum, valid_rows, valid_n)
                    pl.transpose(cur_out_nd, cur_out_dn)
                    pl.store(out, cur_out_nd, [m_off, n_off])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 144, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_vf_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax VF kernel passed!")
```

#### 实现说明

- FP32 VF寄存器包含64个lane。每个AIV最多处理32个M行，`vf.update_mask(valid_rows)`只使能对应lane；64列物理宽度使每个N位置的起始地址按一个完整寄存器对齐。
- 转置前后的UB Tile均声明为64×64。`TTRANS`的实际转置范围来自源Tile的`valid_shape`：源、目标的有效区域分别为`[valid_rows, valid_n]`和`[valid_n, valid_rows]`，VF循环上界和predicate再分别处理N/M尾块。
- `vf.mem_bar(mode=pypto_pro.language.MemBarMode.VST_VLD)`处理VF/Vector store到后续Vector load之间的局部内存依赖，包括跨N块读取`global_max`/`global_sum`，以及`transpose`读取归一化结果。MTE2、MTE3与VF之间的Tile流水依赖仍由`auto_mutex=True`管理。

## 进阶示例：FlashAttention的高性能VF写法

前面的基础示例先将Matmul结果写入GM中的workspace，再由Vector侧读取。高性能FlashAttention实现计算$O=\operatorname{Softmax}(QK^T/\sqrt{D})V$，中间结果通过片上CV通路在AIC和AIV间传递，只有最终输出写回GM。

该实现的主要逻辑如下：

1. Cube侧分块计算$QK^T$，并将L0C Buffer中的结果搬到UB，交给Vector侧处理。
2. Vector侧使用VF完成分块Softmax和在线统计量更新，通过mask处理M、N方向的尾块。
3. Vector侧将Softmax概率从UB写入L1 Buffer，Cube侧再计算$PV$；Vector侧接收结果并完成在线输出更新和最终写回。
4. 多组Tile通过Tile Group轮转，`auto_mutex=True`管理核内Tile访问；AIC/AIV之间使用`set_cross_core`和`wait_cross_core`建立数据就绪与Buffer槽位释放依赖，并通过预取让QK计算与后续PV及Vector阶段重叠。

完整的Kernel、VF辅助函数、片上地址与同步ID配置，以及运行和精度验证代码，请参见[FlashAttention高性能VF实现ST用例](../../../../../../python/tests/st/pypto_pro/frontend/fa/test_fa_perf_tkv_preload_dn_vf_tail_tilegroup.py)。
