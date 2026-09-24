# 性能调优

PyPTO Pro算子的性能受多核任务划分、数据搬运效率、片上数据复用，以及计算与搬运流水并行度等因素影响。性能调优应以采集数据为依据：先分析Kernel耗时、AI Core流水、访存和任务时间线，确定瓶颈所在，再选择相应的优化手段。

性能优化应以功能和精度正确为前提。每次调整Tile Shape、同步关系或多核切分后，均需先通过精度回归，再验证性能收益。

## 性能工具与适用场景

| 工具 | 适用场景 | 主要产物或分析内容 |
|---|---|---|
| torch_npu.profiler | 在Python代码中控制采集范围，适合采集PyPTO Pro Kernel的性能数据。 | Kernel耗时、AI Core流水指标和任务时间线。 |
| [msOpProf](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md) | 单算子上板性能分析和核内流水采集。 | 算子耗时、AI Core流水、内存、Cache、资源冲突和核内流水图。 |

`msOpProf`由CANN包中的`msopprof`可执行程序提供，其命令接口与`msprof op`一致。下文使用`msprof op`命令进行说明。

## 建立性能基线

可比较的性能数据必须来自相同的运行条件。采集前，固定输入Shape、数据类型、TilingKey、`block_dim`、Stream和同步位置，并完成JIT预热。测量纯Kernel耗时时，不要将输入分配、随机数生成、Host到Device拷贝和结果校验计入其中。优化前后使用相同的采集配置，并通过多次测量排除偶然波动。

不同测量口径的耗时不能直接比较。记录结果时需注明测量口径，算子交付前还需验证实际调用链的端到端性能。

## 使用torch_npu.profiler采集

`torch_npu.profiler`可在Python代码中采集PyPTO Pro Kernel的耗时和AI Core流水指标。以下示例假设待测脚本已经定义`kernel`、`block_dim`和`args`：

```python
import torch
import torch_npu


profiler_output = "./profiling_output"
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

# 在正式采集前完成JIT预热
kernel[None, block_dim](*args)
torch.npu.synchronize()

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        profiler_output, analyse_flag=True
    ),
):
    kernel[None, block_dim](*args)
    torch.npu.synchronize()
```

| 文件 | 检查内容 |
|---|---|
| kernel_details.csv | 在Name或Type列中定位目标Kernel，查看Duration(us)、Block Dim及AI Core流水指标。 |
| trace_view.json或trace_result.json | 查看应用层、CANN层和NPU任务的时间关系。 |

## 使用msOpProf采集

使用`msprof op`命令对PyPTO Pro测试脚本中的算子进行上板性能采集：

```bash
msprof op python3 test_example.py
```

正式采集前，先在测试脚本中完成JIT预热。通过`--aic-metrics`选择所需的AI Core指标；采集多组指标时，应保持输入和运行条件不变。核内流水的采集命令和参数配置请参考[《msOpProf用户指南》](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)。常用指标如下：

| 指标 | 主要检查内容 | 可定位的问题 |
|---|---|---|
| PipeUtilization | Cube、Vector、Scalar及数据搬运流水的耗时或占比。 | 主导流水、流水空闲和重叠不足。 |
| ArithmeticUtilization | Cube和Vector计算指令的执行情况。 | 冗余计算或计算单元利用率不足。 |
| Memory、MemoryL0、MemoryUB | GM、L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer和UB等存储层级的读写带宽。 | 搬运受限、有效带宽不足或片上复用不足。 |
| L2Cache | L2 Cache访问和命中情况。 | 数据切分或访问顺序不合理造成的Cache复用不足。 |
| ResourceConflictRatio | UB bank conflict、bank group及其他资源冲突。 | 片上数据排布或并发访问冲突。 |

采集结果保存在`OPPROF_*`目录中。输出文件取决于`--aic-metrics`配置，常用文件如下：

| 文件 | 检查内容 |
|---|---|
| OpBasicInfo.csv | 算子名称、类型、Task Duration(us)、Block Dim、Device ID和AI Core频率等基本信息。 |
| PipeUtilization.csv | 各Core上计算与搬运流水的耗时和占比。 |
| ArithmeticUtilization.csv | Cube和Vector计算指令的耗时和占比。 |
| Memory.csv、MemoryL0.csv、MemoryUB.csv | GM及各级片上存储的读写带宽。 |
| L2Cache.csv | L2 Cache命中情况。 |
| ResourceConflictRatio.csv | UB bank conflict、bank group及其他资源冲突。 |
| trace.json | 各Core内部Scalar、MTE、Cube、Vector和Fixpipe等流水的指令级时间线。使用Trace Viewer或MindStudio Insight打开后，可检查流水空洞、依赖关系和并行情况。 |
| visualize_data.bin | MindStudio Insight使用的可视化数据文件。 |
| dump/ | 采集生成的原始数据、Kernel二进制等中间文件，通常无需直接查看。 |

## 性能数据与流水图分析

采集完成后，按以下顺序定位瓶颈：

1. 从`kernel_details.csv`或`OpBasicInfo.csv`中确认目标Kernel，对比多个样本的Kernel耗时。性能结论以独立、稳定的多轮测量结果为准。
2. 比较Cube、Vector和数据搬运流水的耗时及占比，识别主导流水。流水占比只能反映时间覆盖情况；判断计算效率和带宽利用率时，还需结合指令吞吐、带宽和阻塞指标。
3. 根据计算量、数据搬运量和目标硬件规格估算理论耗时，再与实测结果对比，判断瓶颈来自工作量本身，还是硬件利用率不足。
4. 使用`trace.json`检查关键流水是否连续，以及搬运、Vector和Cube能否有效重叠。
5. 根据分析结果选择优化方向，例如多核切分、Tile Shape、片上复用或多缓冲。每轮只调整一类关键因素，完成精度回归后，再按相同口径复测。

流水优化的目标，是让决定Kernel性能的主流水在稳态阶段连续执行，并尽可能将其他流水的开销隐藏在主流水中。当总耗时主要受某条计算或搬运流水限制时，Kernel即达到该流水的bound状态。矩阵计算类Kernel通常以Cube bound为目标；矢量计算或搬运密集型Kernel则可能分别达到Vector bound或MTE bound。判断bound不能只看累计占比，还要确认主流水在主要执行区间内是否连续，以及其他流水是否与其充分重叠。

图1中的Cube和Vector流水均存在明显间隙，计算任务未能连续执行。此时Kernel尚未达到稳定的计算bound，应重点检查Tile粒度、数据依赖、同步等待，以及搬运与计算的重叠方式。

**图1 Cube和Vector流水存在间隙**

![Cube和Vector流水均存在明显间隙](../../../figures/pro/pro_pipeline_with_gaps.png "Cube和Vector流水均存在明显间隙")

图2中的Cube流水在稳态阶段保持连续，MTE、Fixpipe和Vector任务与Cube计算并行执行。此时Kernel达到Cube bound，即达到Cube流水的性能上限。

**图2 Cube流水达到bound状态**

![Cube流水连续执行并形成Cube bound](../../../figures/pro/pro_pipeline_cube_bound.png "Cube流水连续执行并形成Cube bound")

## 多核切分优化

PyPTO Pro使用[`pypto_pro.language.get_block_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_idx.md)和[`pypto_pro.language.get_block_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md)划分多核任务。切分方案既要完整覆盖计算范围、避免多个Core重复写入同一输出区域，也要保证各Core负载均衡。

- `block_dim`不宜超过可并行执行的任务数，否则会产生空闲工作单元。逻辑核数的取值约束参见[Kernel核函数](../development/kernel_function.md#blockdim的含义与设置)。
- 各Core的工作量应尽量接近，避免将尾块或高开销分支集中到少数Core。
- 规则二维Tile可先线性编号，再按Core编号进行跨步分配，以减小尾部负载差异。
- 输出区域应由唯一Core写入；需要跨Core归约时，应使用明确且受支持的同步与归约方案。
- 小Shape和大Shape应分别选择核数，单一`block_dim`通常无法覆盖所有场景的最优配置。

调试阶段使用`block_dim=1`有助于验证逻辑，但性能测试必须恢复目标核数。

## Tile与片上内存优化

Tile Shape同时影响单次计算量、片上内存占用、循环次数、尾块比例和搬运效率。选择Tile时需综合考虑以下因素：

- 各类片上Buffer的容量限制。
- 数据类型和Tile Shape共同决定的实际字节数。
- 数据搬运和计算指令的对齐要求。
- 尾块比例以及有效数据之外的补齐开销。
- 同一份输入数据在片上的复用次数。
- 多缓冲后总内存占用的倍增。

较大的Tile可以减少循环和指令下发次数，但会占用更多片上内存，可能无法启用双缓冲，甚至降低并行度。较小的Tile资源占用较低，但会增加循环次数、搬运次数和尾块处理开销。最终配置应根据目标Shape实测确定。

GM访问应保持连续和对齐，并尽量合并为大粒度搬运。需要重复使用的数据，在容量允许时应驻留片上，避免反复从GM加载；能够由后续计算直接消费的中间结果，也不应写回GM后再重新加载。

## 流水与多缓冲优化

[`pypto_pro.language.make_tile_group`](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)为同一逻辑数据分配多块轮转Tile：`current()`返回当前Tile，`next()`切换到下一块。启用`@pypto_pro.language.jit(auto_mutex=True)`后，编译器根据Tile的mutex信息插入核内同步，从而构建双缓冲或N缓冲流水。

优化时应重点检查：

- 加载下一块数据能否与当前块的Vector或Cube计算重叠。
- 当前结果写回能否与后续计算重叠。
- 每个轮转Tile是否使用独立且不冲突的mutex编号。
- `next()`的调用次数和位置是否与数据的生产、消费顺序一致。
- 手动同步是否正确配对，是否与自动mutex重复保护同一依赖。
- 流水末尾是否正确排空，最后一个结果是否完成写回。

增加缓冲数量会提高片上内存占用和调度开销。只有双缓冲仍无法隐藏关键延迟时，才考虑增加缓冲级数，并通过Profiling确认收益。

## Cube与Vector协同优化

矩阵计算通过[`pypto_pro.language.section_cube()`](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md)描述Cube任务，矢量计算通过[`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md)描述Vector任务。对于混合Kernel，应尽量重叠Cube计算、Vector前后处理和DMA搬运，并减少不必要的数据格式转换和跨存储层搬运。

- Cube计算应检查M、N、K方向的Tile Shape、左右矩阵布局、转置方式以及L0A Buffer/L0B Buffer装载格式。
- 归约长度较大时，应在L0C Buffer中完成分块累加，再按需要转换并写回。
- Vector前后处理应尽量与Cube流水重叠，避免形成全局串行阶段。
- 连续执行多个细粒度Vector操作且额外开销明显时，可在确认瓶颈后使用Vector Function表达寄存器级计算，减少中间Tile读写。
- 调整Cube与Vector的并行关系后，必须重新检查mutex和实际数据依赖，不能以破坏正确性为代价消除同步。

## TilingKey与编译期特化

TilingKey用于区分数量有限、执行路径差异明显的编译期配置。例如，可为对齐与非对齐路径、不同算法模式或少量固定布局分别生成专用Kernel，消除热循环中的无效分支。

不要将取值范围较大的运行时Shape逐一展开为TilingKey。Key数量过多会增加二进制规模、编译时间、缓存占用及测试成本。只有特化收益明确且候选集合可控时，才适合新增TilingKey。

对于常用Shape，可设计规整、无分支的主路径，将尾块和低频场景放入独立分支。热循环中应减少依赖运行时数据的分支，同时保留必要的边界检查。

## 结果验证与交付检查

完成调优后，应执行以下检查：

- 全部目标Shape、数据类型、布局、TilingKey和`block_dim`均通过精度测试。
- 覆盖最小Shape、非对齐Shape、尾块、空闲Core和最大资源占用等边界场景。
- 在相同测量条件下比较基线与优化版本，并保存多轮结果。
- Profiling数据能够解释性能变化，不以偶然波动作为优化结论。
- 已移除调试接口和仅用于定位问题的额外同步。
- JIT直接调用和实际aclnn调用路径均完成性能验证。
- 二进制大小、TilingKey数量和首次编译时间仍处于可接受范围。
