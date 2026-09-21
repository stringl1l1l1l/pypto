# pypto_pro.language.make_tensor

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

用一个裸指针（pypto_pro.language.Ptr[dtype]）或已有Tensor，加上显式的shape和可选的stride，构造一个Tensor视图。Tensor视图本身不分配内存，只为同一段GM地址设置“形状 + 步长”的解释，之后即可像普通GM Tensor一样被[load](../memory_data_movement/load.md)/[store](../memory_data_movement/store.md)使用。省略stride时，接口根据shape自动生成连续的行主序stride。

常与[pypto_pro.language.addptr](addptr.md)配合：用addptr切出workspace的某一段地址，再用make_tensor把它包装成可读写的Tensor。

下图展示make_tensor的核心语义：它使用shape和stride为已有地址创建Tensor视图，不申请新内存，也不搬运数据。

![make_tensor从裸指针创建Tensor视图](../../figures/make_tensor_view.jpg "make_tensor从裸指针创建Tensor视图")

## 函数原型

```python
pypto_pro.language.make_tensor(
    ptr: Union[Ptr, Tensor],
    shape: Sequence[Union[int, Scalar]],
    stride: Optional[Sequence[Union[int, Scalar]]] = None,
    dtype: Optional[DataType] = None,
) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| ptr | 输入 | 用于构造Tensor视图的地址对象，Ptr或Tensor类型。创建的Tensor视图与源对象共享数据地址。 |
| shape | 输入 | Tensor形状，Sequence[int或Scalar]类型，每一维可以是整型常量或运行时整型Scalar表达式。 |
| stride | 输入 | Tensor步长，Sequence[int或Scalar]类型，可选，单位为元素。显式传入时，长度必须与shape的维数相同，元素[i0, i1, ...]相对ptr的元素偏移为i0 × stride[0] + i1 × stride[1] + ...。省略时自动生成连续行主序stride。对于亚字节数据类型，最内层维必须连续，即最后一项stride必须为1。 |
| dtype | 输入 | Tensor元素类型，[DataType](../basic_data_structures/DataType.md)类型，可选。传入时按该类型解释源地址；省略时沿用源指针或源Tensor的数据类型。 |

## 约束说明

- 创建的Tensor只是地址视图。调用方必须保证shape、stride和dtype描述的所有访问均落在源对象的合法内存范围内，并满足dtype的地址对齐要求。
- stride还用于load和store的起始地址换算及GM搬运描述，必须满足对应搬运接口的能力范围；通用load和store主要用于末维连续的二维搬运。

## 返回值说明

返回与输入ptr共享地址的Tensor视图。

## 调用示例

### 连续行主序Tensor视图

省略stride时，接口自动生成连续的行主序步长：最后一维步长为1，其余各维步长等于后续各维大小的乘积。

```python
normal = pl.make_tensor(ptr, [8, 16])
# 等价于：normal = pl.make_tensor(ptr, [8, 16], [16, 1])
```

### 行间不连续的Tensor视图

下面的Tensor每行包含16个连续元素，相邻两行的起始地址相隔32个元素，因此行间跳过16个元素：

```python
pitched = pl.make_tensor(ptr, [8, 16], [32, 1])
# pitched[i, j]的元素地址：ptr + i * 32 + j
```

### 交换shape和stride表达转置视图

```python
normal = pl.make_tensor(ptr, [8, 16], [16, 1])
transposed = pl.make_tensor(ptr, [16, 8], [1, 16])
```

两者共享同一个ptr，且normal[i, j]与transposed[j, i]指向相同地址；接口只改变逻辑索引到物理地址的映射，不转置原数据。

![make_tensor的非连续与转置视图](../../figures/make_tensor_stride_cases.jpg "make_tensor的非连续与转置视图")

### 在Kernel中访问行间不连续的Tensor

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [256, 1])

    tt = pl.TileType(shape=[32, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])

        pl.load(t, a, [32, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [32, 0])
        pl.load(t, ws_buf, [32, 0])
        pl.store(out, t, [32, 0])
```

### 重建Tensor视图并合并连续维度

pypto_pro.language.make_tensor可以在不搬运数据的情况下，通过shape和stride重建GM Tensor视图，用于调整维数、形状或步长，以及合并连续维度。新Tensor视图与原Tensor共享存储空间，其stride必须与实际内存排布一致。

通过pypto_pro.language.Ptr重建视图时，将Kernel参数声明为pypto_pro.language.Ptr[dtype]，并在函数体内调用pypto_pro.language.make_tensor(ptr, shape, stride)。shape中的各维长度可以来自运行时TilingData，但shape列表的长度必须在编译期确定。

#### 通过Ptr创建Tensor视图

```python
@pl.jit()
def k(q: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    # 行优先 => stride = [n*d, d, 1]；shape/stride 全部来自 tiling
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.n, tiling.d],
                              [tiling.n * tiling.d, tiling.d, 1])
    # 之后 tensor_q 的用法与 pl.Tensor 参数完全相同
```

pypto_pro.language.make_tensor的第一个参数也可以是已有的pypto_pro.language.Tensor。新视图与原Tensor共享地址，并使用新指定的shape、stride和可选dtype。

合轴是将GM Tensor中相邻且连续的两个维度合并为一个维度。与通过order从多维Tensor中选择两个维度不同，合轴会降低Tensor视图的维数，使其与二维Tile对应。典型用途如下：

- 对于FlashAttention的TND或BSND排布，将批次维B和序列维S合并为总token维B × S，即TND中的T维，使一次load可以跨批次连续读取多行。
- 对于维数不同的Host输入，将前若干连续维度合并，在Kernel内构造固定的二维[M, N]视图。

#### 合并连续维度

仅当待合并的维度在内存中连续且不存在间隔时，才能合轴。对于行优先ND排布的[d0, d1, d2]，其stride为[d1 × d2, d2, 1]：

- 合并d0与d1后，新维度大小为d0 × d1，stride取内层维度d1的stride。合并条件为stride(d0) = d1 × stride(d1)。
- 如果d0与d1之间存在Padding，即stride(d0) > d1 × stride(d1)，则不能合并，否则Padding区域会被当作有效数据读取。

#### 将[B, S, D]合并为[B × S, D]

通过pypto_pro.language.make_tensor合并连续维度并构造低一维的Tensor视图，再调用load搬入Tile。

```python
# 原始 GM：q 是 [B, S, D]，行优先连续，stride = [S*D, D, 1]
q: pl.Tensor[[B, S, D], pl.DT_FP16]

# 合轴：B、S 合并成一维 B*S，stride 取里层的 D
q_merged = pl.make_tensor(q, [B * S, D], [D, 1])   # 复用 q 的指针，不搬数据

# 现在 q_merged 是二维 [B*S, D]，直接按 Tile 索引 load
tile = pl.make_tile(pl.TileType(shape=[TS, D], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec), addr=0x0)
pl.load_tile(tile, q_merged, [t, 0])   # 第 t 块 = 合并轴上的第 [t*TS : (t+1)*TS] 行
```

若要读取原始Tensor中第b个批次的第s行，合并后的行号为b × S + s。

#### 使用运行时形状合并为[M, N]

Kernel可以接收指针和TilingData，将Host侧2～4维输入的形状合并为固定的二维Tensor视图。N为最内层维度，M为其余维度的乘积。

```python
@pl.jit(auto_mutex=True)
def add_dynrank_kernel(x: pl.Ptr[pl.DT_FP16], y: pl.Ptr[pl.DT_FP16],
                       z: pl.Ptr[pl.DT_FP16], tiling: AddTiling):
    N = tiling.shape[3]
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]   # 前三维合轴成 M
    tensor_x = pl.make_tensor(x, [M, N], [N, 1])              # 折叠成二维 [M, N]
    tensor_y = pl.make_tensor(y, [M, N], [N, 1])
    tensor_z = pl.make_tensor(z, [M, N], [N, 1])
    ...
    pl.load_tile(tile_a, tensor_x, [i, j])                    # 之后就是普通二维 load
```

逐元素算子只依赖连续的元素顺序，因此可以将[2, 4, 256, 256]、[8, 256, 256]和[512, 512]等不同形状合并为[M, N]，由同一个Kernel处理。Kernel内通过pypto_pro.language.make_tensor构造的视图始终为二维。

#### 将BSND排布转换为TND视图

FlashAttention的TND排布将批次与序列合并为总token维T = ΣS_i。对于形状为[B, S, N, D]的BSND Tensor，如果各批次的S相同且数据在内存中连续，可以合并B、S维，构造TND视图。

```python
# BSND -> 把 B、S 合成 T = B*S（N、D 保留），stride 取里层
q_tnd = pl.make_tensor(q, [B * S, N, D], [N * D, D, 1])
# 再用 order 在 [T, N, D] 里挑 (T, D) 两维
pl.load_tile(q_tile, q_tnd, [t_off, n_idx, 0], order=[0, 2])
```

#### 小结

| 步骤 | 做法 |
|------|------|
| 1. 确认可合轴 | 被合并的两维在内存里连续（无padding）。 |
| 2. 重建视图   | pypto_pro.language.make_tensor(src, 合并后的shape, 合并后的stride)，stride取里层维的stride。 |
| 3. 正常load  | 合并后的Tensor维数更低，按普通二维 / 多维场景load即可。 |
| 4. 偏移换算   | 合并轴的行号 = 外层下标 * 里层大小 + 里层下标。 |

> **合轴约束**：被合并的维度必须连续，stride必须与实际内存排布一致。违反上述约束会读取错误位置或Padding区域的数据。排查精度问题时，可改用未合轴的逐维Tensor视图进行对比验证。
