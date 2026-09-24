# 功能调试

PyPTO Pro Kernel出现功能问题时，先固定可复现的输入Shape、数据类型、TilingKey和`block_dim`，再根据现象选择定位路径：首次调用时编译报错，查看前端异常和错误码；Kernel正常执行但结果不正确，定位精度问题；执行时出现AIC Error、超时或挂死，采集设备侧异常现场。内存检测是排查疑似内存访问问题的工具，可在后两类问题中按需开启。

问题修复后，应关闭内存检测并移除临时调试打印，使用生产配置重新编译，覆盖原失败用例及相关Shape、数据类型、TilingKey和`block_dim`进行回归。

## 编译报错

被`@pypto_pro.language.jit()`装饰的Kernel在需要生成新编译实例时依次完成参数绑定、前端解析与校验、代码生成和编译。首先查看异常中的用户源码位置、错误描述和调用栈，判断失败环节。参数绑定、Python前端解析或IR校验失败时，重点检查接口参数、Shape、数据类型、内存空间及编译期约束。

如果报错带有PyPTO错误码，可结合[错误码参考](../../../appendix/trouble_shooting/)判断所属组件并查阅处理建议。报错位置和错误描述仍是定位用户代码问题的直接依据。代码生成或工具链编译失败时，还应检查完整编译日志和生成代码。

JIT编译产物默认位于当前工作目录下的`build`目录；设置`ASCEND_WORK_PATH`后，位于`${ASCEND_WORK_PATH}/PYPTO_PRO/build`目录。可重点查看以下文件：

| 文件 | 用途 |
| --- | --- |
| `kernel.cpp` | 编译器生成的设备侧Kernel源码，可用于检查参数类型、控制流、地址计算和数据搬运。 |
| `pipeline_generated.py` | 启用[自动CV并行流水](../advanced_programming/auto_parallel_pipeline.md)后生成的自动核间流水排布代码，可用于检查stage排布、preload和核间同步。 |

以上文件均为编译器生成的诊断产物。定位到问题后，应修改原始Python Kernel并重新编译，不要直接修改生成文件。

## 精度错误

当Kernel能够正常执行，但输出与参考实现不一致时，先确认输入、Shape、数据类型、布局和比较阈值一致，并在设备同步后读取结果。固定失败用例，再按[pypto-pro-precision-debug Skill](https://gitcode.com/cann/pypto-gym/blob/master/cannbot-skills/ops/pypto-pro-precision-debug/SKILL.md)提供的流程，沿实际数据路径查找第一个出错环节。

需要检查多核切分、循环边界、运行时参数或搬运偏移时，可使用[`pypto_pro.language.printf`](../../../../api/pro_api/Utils-API/debugging/printf.md)打印Core编号、偏移、Shape和分支标志等标量信息。多核场景建议通过`pypto_pro.language.get_block_idx()`限制打印的Core，避免日志大量交错。该接口会引入运行时开销，仅用于功能调试，不应用于性能测试。

如果结果错乱、偶发错误，或怀疑GM访问越界、Tile访问越界、Tile内存区间重叠、mutex未正确配对，可在核函数装饰器上设置`@pl.jit(sanitizer=True)`，使用[内存检测工具](sanitizer.md)辅助定位。检出问题时，结果写入报告文件；工具的覆盖范围和使用约束详见该文档。内存检测会产生额外的运行时开销，完成排查后应关闭。

## AI Core Error

Kernel下发是异步操作。调试时应在启动后同步Stream，使设备侧异常在当前调用点暴露，并保留完整的报错信息和运行日志：

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()
```

Kernel在设备上触发越界读写、死锁超时等AIC Error后，可通过异常dump和离线复现工具还原现场，并将Error PC定位到Kernel源码行。执行用例前设置一个可写的工作目录：

```bash
export ASCEND_WORK_PATH=./wk
```

采集前请确保未设置`NPU_COLLECT_PATH`；如果已经设置，可执行`unset NPU_COLLECT_PATH`将其取消，否则无法生成离线复现所需的数据文件。建议在独立子进程中运行问题用例，以确保异常数据完整生成。

异常发生后，相关文件保存在以下目录：

```text
${ASCEND_WORK_PATH}/extra-info/data-dump/<device_id>/
```

主要包括：

| 产物 | 用途 |
| --- | --- |
| dump数据文件 | 保存输入、输出和workspace Tensor的Shape、数据类型及原始数据。 |
| `<kernel>_launch_args.json` | 保存Kernel名称、`block_dim`以及完整的指针和标量启动参数。 |
| `<kernel>_call_kernel.so` | 带调试信息的Kernel执行体，用于离线复现和源码行定位。 |
| `call_kernel_<digest>.so` | 原始JIT产物副本，用于比对和高保真复现。 |
| `kernel.cpp`、`call_kernel.cpp`和相关头文件 | 异常Kernel对应的源码副本。 |

使用以下命令解析dump并复现异常：

```bash
python tools/scripts/debug_aicore_error_pro_repro.py \
    -p <dump_dir> \
    [-d <device_id>] \
    [-out <output_dir>] \
    [-t <seconds>]
```

其中，`-p`指向`extra-info/data-dump/<device_id>`目录；`-d`用于覆盖从目录名解析出的Device ID；`-out`用于指定报告输出目录；`-t`用于设置单次复现的超时时间，默认值为600秒。

工具会恢复Tensor数据和Kernel启动参数，生成并执行单算子复现脚本，再从本次运行的plog中提取Error PC。主要输出包括：

- `test_single_op.py`：可独立执行的单算子复现脚本。
- `reproduction_report.txt`：复现结果、出错Core、Error PC、符号、源码行、inline调用链和源码上下文。
- `reproduction_report--singlecommit.txt`：超时类故障的singlecommit诊断结果。仅超时类故障生成。

对于超时类故障，工具会自动在singlecommit模式下重跑：若singlecommit下执行成功，说明问题与核间并发或同步时序有关；若仍然失败，则应结合报告中的卡死Core和源码位置继续排查。

若报告提示`Error PC not found in plog`，应检查CANN是否生成了symbol locator日志及日志目录配置。若报告提示源码行无法解析，应检查`ASCEND_HOME_PATH`和BiSheng工具链是否可用。旧dump缺少`<kernel>_launch_args.json`时，工具会将`block_dim`回退为`1`，依赖多核并发的问题可能无法按原现场复现。

若根据异常位置怀疑内存访问问题，可尝试开启[内存检测工具](sanitizer.md)辅助排查。工具未检出问题时，还应结合异常报告继续检查同步、地址计算和Kernel源码。
