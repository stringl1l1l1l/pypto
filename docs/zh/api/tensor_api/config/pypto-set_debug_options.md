# pypto.set\_debug\_options

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

设置debug的选项。

## 函数原型

```python
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  dump_pass_graph: Optional[List[str]] = None,
                  ) -> None
```

## 参数说明

| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| compile_debug_mode   | 输入      | 含义：设置编译阶段调试模式 <br> 说明：0：代表默认不使能编译阶段调试模式； <br> 1：代表使能图编译阶段调试模式，一键开启图编译相关配置，当前仅包括计算图； <br> 2：代表使能固定CCE功能，一键开启固定设备侧代码输出路径、不覆盖已存在的CCE文件；并行编译线程数仍由 `codegen.parallel_compile` 控制，kernel 名按任务身份确定性编号，多线程编译产物可复现； <br> 类型：int <br> 取值范围：0、1 或 2 <br> 默认值：0 <br> 影响Pass范围：NA |
| runtime_debug_mode   | 输入      | 含义：设置执行阶段调试模式 <br> 说明：0：代表默认不使能执行阶段调试模式； <br> 1：代表使能执行阶段调试模式，一键开启图执行相关配置，当前仅包括泳道图； <br> 2：代表使能 AICORE_MODEL 仿真模式； <br> 3：代表使能运行时依赖正确性校验数据 dump； <br> 4：代表使能运行时 GM 内存越界检查，并在控制流入口对每个由用户显式登记并规范化后的 `assume_divisible` 假设表达式生成运行时整除断言；生成代码使用实际运行时值计算表达式，未显式登记的编译器推导表达式不检查； <br> 类型：int <br> 取值范围：0、1、2、3 或 4 <br> 默认值：0 <br> 影响Pass范围：NA |
| dump_pass_graph      | 输入      | 含义：指定一个或多个Pass开启计算图dump和IR打印 <br> 说明：空列表表示不启用；配置为Pass标识符列表时，仅对匹配Pass开启计算图dump和IR打印，其它Pass不输出计算图和IR。若配置了不存在的Pass标识符，该标识符不会命中任何Pass，并输出告警。 <br> 类型：List[str] <br> 默认值：[] <br> 影响Pass范围：指定Pass |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

- 当 compile_debug_mode=2（固定CCE）时，设备侧代码输出路径由环境变量 ASCEND_WORK_PATH 决定（`$ASCEND_WORK_PATH/pypto/<name>`）；若未设置 ASCEND_WORK_PATH，则输出到当前工作目录。此模式下 TILE_FWK_OUTPUT_DIR 不参与设备侧代码路径计算。
- compile_debug_mode=2 等效于同时开启 `codegen.fixed_output_path` 并关闭 `codegen.force_overwrite`；已存在的 CCE 文件不会被覆盖，便于在固定路径下反复修改 CCE 调试。
- dump_pass_graph 配置项填写的是当前Pass策略中的Pass标识符（identifier），例如"RemoveRedundantReshape"，不是C++中的PassName枚举值。
- dump_pass_graph 支持配置多个Pass标识符；列表中未匹配当前Pass策略的标识符会被忽略，并输出告警。若列表中所有标识符均未匹配，则不会输出任何Pass的计算图或IR。
- 当 compile_debug_mode=1 与 dump_pass_graph 同时设置时，compile_debug_mode=1 优先，最终开启所有Pass的计算图dump和IR打印。

## 调用示例

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
pypto.set_debug_options(compile_debug_mode=2)
pypto.set_debug_options(dump_pass_graph=["RemoveRedundantReshape", "InferMemoryConflict"])
```
