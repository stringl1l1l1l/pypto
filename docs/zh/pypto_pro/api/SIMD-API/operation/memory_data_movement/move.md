# pypto_pro.language.move

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

在L1、L0A/L0B、L0C、UB等各级内存之间搬运tile（如L1→L0A/L0B、L0C(Acc)→UB(Vec)、UB→L1等）。具体走哪条硬件搬运通路、用哪条流水，由**源与目的tile的内存空间**决定。

可在搬运的同时融合ReLU、预量化，或经fixpipe做量化（`scale`）；也可通过`offset`从一块较宽的源tile中提取子块（对应后端`pto::TEXTRACT`）。

当带`offset`且源tile是目的tile的子块（目的每一维均不小于源，至少一维严格大于）时，`move`自动转为[`insert`](insert.md)语义（把较小的源tile嵌入较大的目标tile），无需手动调用`insert`。当源/目的layout互为ZN/NZ（物理转置）时，按转置后的shape比较。

`insert`作为显式入口仍保留，但`move`已覆盖其能力。注意：带了`move`专属参数（`acc_to_vec_mode`/`relu_pre_mode`/`scale`）时不自动转insert，仍按`move`语义处理。

与[`pypto_pro.language.load`](load.md)/[`pypto_pro.language.store`](store.md)（tensor↔tile，跨GM）不同，`move`是tile↔tile，不涉及GM。

## 函数原型

```python
pypto_pro.language.move(dst_tile, src_tile, offset=None, *, acc_to_vec_mode=None,
        relu_pre_mode=None, scale=None, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 目标tile，搬入目的地；其内存空间决定TMOV变体 |
| `src_tile` | 输入 | 源tile |
| `offset` | 输入 | 可选，`[offset_m, offset_k]`；源tile不小于目的tile时为提取子块（TEXTRACT），源tile为目的tile的子块时自动转为嵌入（TINSERT，见功能说明） |
| `acc_to_vec_mode` | 输入 | 可选，Acc→Vec搬运模式 |
| `relu_pre_mode` | 输入 | 可选，搬运时融合ReLU |
| `scale` | 输入 | 可选，随路量化比例（deqScalar / deqTensor路径） |
| `phase` | 输入 | 可选，详见 [phase 使用约束](../matrix_computation/phase.md) |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>内存空间与源共同决定流水（见"流水类型"）；首地址必须32字节对齐 |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>提取子块（TEXTRACT）时源tile须不小于目的tile；自动转为嵌入（TINSERT）时源tile须为目的tile的子块 |
| `offset` | 输入 | 二维`[offset_m, offset_k]`，单位为元素个数<br>**TEXTRACT（源≥目的）**：以源tile声明的物理`shape`为坐标系；`valid_shape`不改变offset的坐标原点和计量单位，每一维须满足`0 <= offset < src_tile.shape`<br>实际提取范围为`[offset, offset + dst_tile.valid_shape)`。使用尾块时，完整的`dst_tile.shape`可以超出offset后的剩余范围，但实际提取范围不得超出`src_tile.shape`；若源tile设置了更小的`valid_shape`，实际提取范围还须位于该有效区域内<br>**TINSERT（源<目的，自动转insert）**：改以目的tile为坐标系，须满足`offset[0] + src行数≤ dst行数`、`offset[1] + src列数≤ dst列数`，否则越界 |
| `acc_to_vec_mode` | 输入 | 取`pl.AccToVecMode.SingleModeVec0`/`pl.AccToVecMode.SingleModeVec1`/`pl.AccToVecMode.DualModeSplitM`/`pl.AccToVecMode.DualModeSplitN`；仅在源为`Acc`、目的为`Vec`时有意义。`scale`为`Tile`（per-channel）时只支持单vec模式（`DualModeSplitM`/`DualModeSplitN`报错） |
| `relu_pre_mode` | 输入 | 默认`None`（不融合ReLU）；可取`pl.ReluPreMode.NormalRelu` |
| `scale` | 输入 | 可选，随路量化比例：`float`（编译期标量）→ per-tensor量化；运行时`FP32`标量→自动重解释为IEEE-754位模式；运行时`INT`标量→须传预编码的float32位模式（`struct.pack("!f", v)`）；`Tile`（INT64、`MemorySpace.Scaling`、shape `[1, N]`）→ per-channel量化（`move_fp`路径），用户预制deqTensor tile，框架直接复用（不自动分配/同步，用户负责load→move→sync(MTE1→FIX)），只支持单vec模式；`Tensor`不支持（per-channel须以`Tile`传入） |
| `phase` | 输入 | 默认`None`；可取`pl.STPhase.Partial`或`pl.STPhase.Final`，仅对源为`Acc`、目的为`Vec`时有效；启用后通过硬件unit_flag与matmul生产者做握手同步，替代软件mutex；不能与`offset`同时使用。详见 [phase 使用约束](../matrix_computation/phase.md) |

## 流水类型

由源/目的内存空间决定：

| 源 → 目的 | 流水 |
|---|---|
| Acc(L0C) → Vec(UB) | FIX（fixpipe） |
| Mat(L1) → Left/Right(L0A/L0B) | MTE1 |
| Mat(L1) → Vec(UB) | V |
| Mat(L1) → 其他 | FIX |
| Vec(UB) → Mat(L1) | MTE3 |
| 其余 | V |

## 调用示例

下面是一个完整matmul kernel：`pypto_pro.language.load`把左右矩阵从GM载入L1，`pypto_pro.language.move`再把L1数据搬到L0A/L0B供cube计算——`move`在此承担L1→L0A/L0B这一步。cube kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def matmul_move_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    tt_right = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    tt_acc = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)

    a_l1 = pl.make_tile_group(type=tt_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_mat, addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_c_l0c = c_l0c.current()
        pl.load(cur_a_l1, a, [0, 0])     # GM -> L1
        pl.load(cur_b_l1, b, [0, 0])
        pl.move(cur_a_l0a, cur_a_l1)            # L1 -> L0A
        pl.move(cur_b_l0b, cur_b_l1)            # L1 -> L0B
        pl.matmul(cur_c_l0c, cur_a_l0a, cur_b_l0b)
        pl.store(out, cur_c_l0c, [0, 0])     # L0C -> GM（源在 Acc，走 FIX 流水）
```

其他典型用法（节选）：

```python
# L0C(Acc) → UB(Vec)，搬运时随路量化（缩放2.0）
pl.move(vec_tile, acc, scale=2.0, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)

# 从宽源 tile 提取子块（TEXTRACT）
pl.move(cur_a_left, a_wide_slot, offset=[0, KL0])

# 源是目的子块，自动按insert语义处理（TINSERT）——无需手动调用pl.insert
pl.move(p_mat, tile_nz, offset=[0, 0])
```

## pl.AccToVecMode.DualModeSplitM / pl.AccToVecMode.DualModeSplitN尾块处理

### 硬件约束

- **pl.AccToVecMode.DualModeSplitM**：对M轴进行切分，硬件会往每一块Vec写`M/2 * N`数据，其中M必须是2的倍数。
- **pl.AccToVecMode.DualModeSplitN**：对N轴进行切分，硬件会往每一块Vec写`M * N/2`数据，其中N必须是32的倍数。

这里的M和N指的是元素个数，与数据类型无关。

从GM→L1→L0A/L0B→L0C的过程中，所有tile的`valid_shape`始终使用**实际尾块大小**。在L0C→Vec的`move`搬运阶段，如果使用`pl.AccToVecMode.DualModeSplitM`或`pl.AccToVecMode.DualModeSplitN`，框架会**自动**在move之前对L0C的`valid_shape`做向上对齐，用户无需手动设置对齐后的值，但需要了解在Vec侧的切分策略：

- **M轴切分策略**：M轴向上对齐到2的倍数，Vec0（sub_id=0）得到前`aligned_M / 2`行，Vec1（sub_id=1）得到剩余`原始M - aligned_M / 2`行。例如M=33，框架对齐到34，Vec0得到17行，Vec1得到33-17=16行。
    - 当尾块M = 1时，Vec0直接取实际尾块大小，Vec1为0。
- **N轴切分策略**：N轴向上对齐到32的倍数，Vec0（sub_id=0）得到前`aligned_N / 2`列，Vec1（sub_id=1）得到剩余`原始N - aligned_N / 2`列。例如N=33，框架对齐到64，Vec0得到32列，Vec1得到33-32=1列。再例如N=65时，框架对齐到96，Vec0得到48列，Vec1得到65-48=17列。
    - 当尾块N ≤ 16时，Vec0直接取实际尾块大小，Vec1为0。

### 总体流程

```text
GM → L1 (load)     : valid_shape = 实际尾块大小 (如 33)
L1 → L0A/L0B (move): valid_shape = 实际尾块大小 (如 33)
L0A/L0B → L0C (matmul): Acc valid_shape = 实际尾块大小 (如 33)
L0C → Vec (move)   : 框架自动对齐 Acc valid_shape（用户无需感知）

Vec侧              : Vec0/Vec1 valid_shape = 切分后的实际份额（用户需自行计算）
```

### pl.AccToVecMode.DualModeSplitM尾块（M轴按2对齐对半切分）

`pl.AccToVecMode.DualModeSplitM`要求M为偶数。尾块M为奇数时，框架在`move`之前自动将Acc的`validRow`向上对齐到2的倍数。**用户无需手动对齐。**

- **框架自动对齐**：`validRow`从`valid_m`变为`(valid_m + 1) / 2 * 2`（如33 → 34）
- **V侧切分计算**：`v0 = (valid_m + 1) // 2 * 2 // 2`（如34 → v0=17），`v1 = valid_m - v0`（如33-17=16）

```python
# cube section: 用户只需设置实际 valid_m，无需手动对齐
pl.set_validshape(ac, [valid_m, N])       # valid_m=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # 框架自动对齐

# vector section: V 侧用户自行计算 Vec0/Vec1 中 Vec 实际大小
v0 = (valid_m + 1) // 2 * 2 // 2          # v0 = 17
v1 = valid_m - v0                         # v1 = 16
if sub_id == 0:
    pl.set_validshape(vec, [v0, N])
else:
    pl.set_validshape(vec, [v1, N])
```

### pl.AccToVecMode.DualModeSplitN尾块（N轴按32对齐切分）

`pl.AccToVecMode.DualModeSplitN`要求N为32的倍数。尾块N不满足时，框架在`move`之前自动将Acc的`validCol`向上对齐到32的倍数。**用户只需在matmul前设置实际valid_n，无需手动对齐。**

- **框架自动对齐**：`validCol`从`valid_n`变为`(valid_n + 31) / 32 * 32`（如33 → 64）
- **V侧切分计算**：`v0 = (valid_n + 31) // 32 * 32 // 2`（如64 → v0=32），`v1 = valid_n - v0`（如33-32=1）
- **尾块≤ 16的特殊情况**：当尾块N ≤ 16时，`v0`直接取实际尾块大小（`v0 = valid_n`），`v1 = 0`。

```python
# cube section: 用户只需设置实际 valid_n，无需手动对齐
pl.set_validshape(ac, [TILE, valid_n])     # valid_n=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)  # 框架自动对齐

# vector section: V 侧用户自行计算 v0/v1
v0 = (valid_n + 31) // 32 * 32 // 2        # v0 = 32
v1 = valid_n - v0                          # v1 = 1
if sub_id == 0:
    pl.set_validshape(vec, [TILE, v0])
else:
    pl.set_validshape(vec, [TILE, v1])
```

### 尾块切分规则总结

| 模式 | 框架自动对齐 | Vec0份额 | Vec1份额 | 非对齐尾块示例（M/N=33） | 对应用例 |
|------|------------|----------|----------|---------------------|---------|
| `pl.AccToVecMode.DualModeSplitM` | M向上对齐到2的倍数 | `aligned_M // 2` | `原始M - v0` | M=33→对齐34→v0=17, v1=16 | `test_split_m_odd_tail` |
| `pl.AccToVecMode.DualModeSplitN` | N向上对齐到32的倍数 | `aligned_N // 2` | `原始N - v0` | N=33→对齐64→v0=32, v1=1 | `test_split_n_odd_tail` |

> **注意**：
>
> - GM→L1→L0A/L0B→L0C全程使用**实际尾块大小**的`valid_shape`，确保matmul只计算有效数据。
> - L0C→Vec的`move`时，框架自动对Acc的`valid_shape`做对齐，用户无需手动设置对齐值。
> - V侧（vector section）用户需自行计算Vec0/Vec1并设置Vec的`valid_shape`，框架不自动处理。
> - 切M轴时，若M=1，则Vec0 = 1，Vec1 = 0。
> - 切N轴时，若N<=16，则Vec0 = N，Vec1 = 0。
