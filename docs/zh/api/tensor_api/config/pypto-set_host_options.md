# pypto.set\_host\_options

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

该接口是编译框架提供的运行时动态配置管理功能的核心部分，它将原本静态配置在tile\_fwk\_config.json中的参数转变为动态、可编程的指令，主要功能是控制上板流程的执行。

## 函数原型

```python
set_host_options(*, compile_stage: Optional[CompStage] = None,
                    compile_monitor_enable: Optional[int] = None,
                    compile_timeout: Optional[int] = None,
                    compile_timeout_stage: Optional[int] = None,
                    compile_monitor_print_interval: Optional[int] = None) -> None
```

## 参数说明

| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | 输入      | 含义：控制编译执行的阶段 <br> 说明：<br> ALL_COMPLETE: 无影响，正常编译与运行; <br> TENSOR_GRAPH: 编译阶段，生成最终张量图后停止; <br> TILE_GRAPH: 编译阶段，生成最终分片图后终止；<br> EXECUTE_GRAPH: 编译阶段，生成最终执行图后终止；<br> CODEGEN_INSTRUCTION: 编译阶段，生成指令代码后终止；<br> CODEGEN_BINARY: 编译生成代码二进制后终止,编译阶段结束。 <br> 取值范围: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY) <br> 默认值: ALL_COMPLETE |
| compile_monitor_enable    | 输入      | 含义：控制编译阶段监控模式（Watchdog 机制）<br> 说明：<br> 0: 关闭Compiler Monitor;<br> 1: 开启Compiler Monitor，关闭Pass明细。编译开始时在INFO日志中记录超时阈值，编译总耗时超过阈值时在WARNING日志中记录超时摘要;<br> 2: 开启Compiler Monitor，开启Pass明细。在模式1行为基础上，编译过程中还会在屏幕打印Pass进度，并在阶段超时时打印WARNING。<br> 取值范围: int [0, 2]<br> 默认值: 1 |
| compile_timeout    | 输入      | 含义：设置编译总耗时超时阈值，当前编译的总耗时超过该值后，打印超时告警提示信息 <br> 说明：仅在compile_monitor_enable为1或2时生效，单位（秒），数值为0时表示禁用告警提示打印 <br> 数值类型: int 。 <br> 取值范围: int [0, 2147483647] <br> 默认值: 600 |
| compile_timeout_stage    | 输入      | 含义：设置编译单阶段耗时超时阈值，编译流程单个阶段的耗时超过该值后，打印超时告警提示信息 <br> 说明：仅在compile_monitor_enable为1或2时生效，单位（秒），数值为0时表示禁用告警提示打印 <br> 数值类型: int 。 <br> 取值范围: int [0, 2147483647] <br> 默认值: 0（禁用） |
| compile_monitor_print_interval    | 输入      | 含义：设置编译进度打印间隔，编译流程单个阶段的耗时超过60s后，开始按照此间隔进行进度打印 <br> 说明：仅在compile_monitor_enable为1或2时生效，单位（秒） <br> 数值类型: int 。 <br> 取值范围: int [0, 2147483647] <br> 默认值: 60 |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

- 类型安全：需要确保传入的参数值类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH,
                       compile_monitor_enable=1,
                       compile_timeout=120,
                       compile_timeout_stage=30,
                       compile_monitor_print_interval=20
                       )
```
