# pypto.experimental.set\_runtime\_options

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

该接口用于设置**实验性运行时配置**。它将`tile_fwk_config.json`中运行时里尚未稳定的参数，转变为可编程接口。后续新增的运行时实验特性也通过本接口扩展，不进入`pypto.frontend.jit(runtime_options=...)`的稳定参数表。

## 函数原型

```python
set_runtime_options(
    *,
    stitch_function_num_per_pool: Optional[list[int]] = None,
)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| stitch_function_num_per_pool | 输入 | 含义：用于分别控制Workspace中三个子内存池的容量大小，格式为`[root_inner, assemble_outcast, exclusive_outcast]`。配置的大小决定每个内存池可容纳的最大stitch构建loop迭代的数量。<br> 说明：根据算子运行时各个子内存池实际使用情况分别精细控制预留的容量大小，可通过该配置达到在满足目标性能时降低内存，或者在优先内存使用时提升性能的目的。<br> 配置方法：`[0, 0, 0]`表示关闭；任一元素非0即开启。开启后，某一维为0表示不为该子内存池预留容量。若算子存在该内存池对应类型的中间Tensor，编译会因容量不足失败，须将该维至少配为1；仅当算子没有该类型Tensor时才可配0。三个子内存池含义及对应的Tensor类型见[示例1](#stitch_function_num_per_pool_detail)。<br> 使用场景：<br> （1）并行度已满足预期，需要进行精细化内存控制，以达到保证性能的同时又降低内存的目的：先用`stitch_function_max_num`将并行度调至目标性能，再按推荐值分别下调三个子内存池，降低内存使用量。<br> （2）优先内存场景下，期望通过精细化内存控制，提升性能：先用`max_workspace_kb`控制内存总量，再精细化控制真正限制stitch构建loop迭代数量的子内存池大小，提升性能。<br> 具体调参步骤见[调优方法](#workspace_tuning_method)。<br> 类型：list of int，固定3个元素<br> 取值范围：每个元素 0～1024<br> 默认值：`[0, 0, 0]`<br> 影响Pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置成功即生效。

## 约束说明

- 类型安全：须为长度为3的list或tuple，元素为`[0, 1024]`内的整数，不能使用bool。
- 作用范围：不要在`pypto.loop`内或kernel内设置。
- 配置项相对关系：`stitch_function_max_num` 在不考虑内存占用情况下只考虑调优性能情况下使用；`max_workspace_kb`在有内存限制时按内存总量压缩内存时使用，可能降低并行度。精细化配置在上述两项之后使用，按三个子内存池分别设置。开启后`stitch_function_max_num`与`max_workspace_kb`均失效。

## 调用示例

```python
pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])
```

### 示例1: 精细Workspace模式 <a id="stitch_function_num_per_pool_detail"></a>

以下示例用于说明默认与精细控制的差异：

```python
B_STATIC, L_STATIC, H_STATIC, D_STATIC = 1, 64, 1, 16

pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])

@pypto.frontend.jit
def k_tmp_to_d_emb(
    dy: pypto.Tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    output1: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
    output2: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
):
    tmp_assemble = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output1.dtype, "tmp_assemble")
    tmp_exclusive = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output2.dtype, "tmp_exclusive")

    # Loop0：Exclusive write
    for i_idx, t in pypto.loop_unroll(0, 1, 1, name="l_loop_0"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        tmp_exclusive[:] = pypto.add(dy, dy)

    # Loop1：Assemble write
    for j_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_1"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, j_idx : j_idx + t, 0]
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dx = pypto.matmul(dy_v, weight[0], pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp_assemble[0, j_idx : j_idx + t, 0] = dx + 0.0

    # Loop2：Read
    for k_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_2"):
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        output1[0, k_idx : k_idx + t] = tmp_assemble[0, k_idx : k_idx + t, 0]
        output2[0, k_idx : k_idx + t] = tmp_exclusive[0, k_idx : k_idx + t, 0]
```

该示例中的tensor内存分布如下：

| Tensor | 数据归属 |
| --- | --- |
| dy、weight、output1、output2 | 输入参数，不进Workspace内存池 |
| dy_v | dy的切片视图，复用同一块内存 |
| dx | RootInner |
| tmp_assemble | Assemble outcast |
| tmp_exclusive | Exclusive outcast |

将`stitch_function_num_per_pool`配置为非全零后，三个子内存池可容纳的stitch构建的loop数量相互独立，可分别按实际需求设置。对本示例，运行时根据实际占用得到的推荐配置为`[64, 1, 1]`，说明如下：

- **root_inner**：由默认的128下调为64。`dx`只存在于loop1的单次循环内，无需为loop0、loop2预留容量。
- **assemble_outcast**：由默认的128下调为1。`tmp_assemble`在循环外创建，多次循环共享同一块内存。
- **exclusive_outcast**：由默认的128下调为1。`tmp_exclusive`仅在loop0的一次循环中产生。

经过上述设置，在并行度基本不变的前提下降低workspace占用。

## 调优方法 <a id="workspace_tuning_method"></a>


### 采集日志

```bash
export ASCEND_PROCESS_LOG_PATH=./wk
export ASCEND_GLOBAL_LOG_LEVEL=1
python your_test.py
```

```bash
grep -rE "\[Workspace Runtime (Pool|Tuning|Summary|Recommendation)\]" ./wk
```

| 日志标识 | 说明 | 重点字段 |
|----------|------|----------|
| `[Workspace Runtime Pool]` | 单次任务提交时各池实际使用 | `current`、`peak`、`capacity` |
| `[Workspace Runtime Tuning]` | 单次任务提交的配置与推荐 | `taskId`、`stitchCount`、`configuredDepths`、`recommendedDepths` |
| `[Workspace Runtime Summary]` | 整次执行各池的最大实际占用与容量上限 | 各池`peak`、`capacity` |
| `[Workspace Runtime Recommendation]` | 整次执行结束后的配置建议，以及某个内存池配置值加1时增加的内存 | `recommendedDepths`、`rawPerParallelBytes`、`nextDepthTotalBytes` |

`recommendedDepths={rootInner, assembleOutcast, exclusive}`按顺序对应`stitch_function_num_per_pool`的三个元素。`actualDepths`为当前各个内存池生效大小。

### 场景1：并行度已满足预期，精细化控制以降低内存

目标：并行度已达到预期时，去掉已申请但并未使用的内存，降低workspace，同时保持性能。

操作步骤：

1. 仅使用`stitch_function_max_num`（或不设置，默认128）将端到端性能调至目标，不要同时开启本配置或`max_workspace_kb`。
2. 按上文开启INFO日志并执行算子。若出现`[Workspace Runtime Alloc Failure]`，说明并行度已被内存截断，须先增大`stitch_function_max_num`或改走场景2。
3. 读取`[Workspace Runtime Recommendation]`的`recommendedDepths`，写入本配置。例如`recommendedDepths={rootInner=64, assembleOutcast=1, exclusive=1}`对应`[64, 1, 1]`。
4. 去掉配置`stitch_function_max_num`后复测：workspace应下降，端到端耗时相对步骤1应基本持平。若性能下降，查看`[Workspace Runtime Summary]`中实际占用已接近容量上限的内存池，将该元素加1后重试。
5. 覆盖算子实际使用的shape与执行路径，对各样本的`recommendedDepths`逐项取最大值后固化，再关闭INFO日志验收。


### 场景2：优先内存，精细化控制以提升并行度

目标：在总内存存在使用限制时，精细控制子内存池大小，把内存用到真正限制stitch构建的loop数量的池上。

`max_workspace_kb`按同一规模压缩三个子内存池，推荐值反映的是当前内存限制下根据实际使用内存反推的并行度，不是内存充足时的最优并行度，不能保证内存是充分利用的。内存未用满时，可按下面步骤提高并行度。

操作步骤：

1. 使用`max_workspace_kb`（须大于提示的最小可运行值）。例如限额200MB时配置`max_workspace_kb=200*1024`。
2. 按上文采集日志。从`[Workspace Runtime Summary]`对比各池实际占用与容量上限，从`[Workspace Runtime Recommendation]`读取`recommendedDepths`与`nextDepthTotalBytes`。
3. 将`recommendedDepths`写入本配置作为起点，并取消`max_workspace_kb`。
4. 计算剩余内存：限额减去当前workspace占用。某个内存池的配置值再加1，整次执行增加的字节数为`nextDepthTotalBytes`中对应项。
5. 优先增加实际占用更接近容量上限、且`nextDepthTotalBytes`能被剩余内存覆盖的池（常见为第1个元素`root_inner`）。每次只将该元素加1，复测端到端耗时与workspace。
6. 当再加1将超过限额，或耗时不再下降时停止。多shape时对各元素取最大值后固化。

示例：限额200MB，实际峰值约100MB，`recommendedDepths={rootInner=32, assembleOutcast=1, exclusive=1}`。剩余约100MB可用于提高`root_inner`，可增加的次数不超过剩余字节除以`nextDepthTotalBytes.rootInner`。提高后workspace上升、并行度变大；若出现`[Workspace Runtime Alloc Failure]`或性能回退，将该元素减回。
