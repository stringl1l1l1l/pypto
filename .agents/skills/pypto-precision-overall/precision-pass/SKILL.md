---
name: precision-pass
description: Pass精度校验技能。开启PreCheck/PostCheck进行全链路Pass校验，通过pass校验定位报错Pass，使用pass_compare逐Op对比定位具体问题Op，支持动态shape上板数据打印验证。触发词：pass精度、pass校验、pass_verify、pass_compare、精度校验。
---
# Pass 精度校验

> **定位到问题归属（Pass 引入偏差 / 同步 / VF / 合轴 / Mix 合图）即终止，不延伸分析框架内部根因。仅反馈归属结果，不分析 CCE 文件、Pass 源码等框架实现。**

验证 PyPTO Pass 侧精度问题，通过 PreCheck/PostCheck 全链路校验定位报错 Pass，使用 pass_compare 逐 Op 对比定位具体问题 Op。

## 总览

```
配置 PreCheck/PostCheck + verify_options → 编译运行 → 检查 interpreter.log
       │
       ▼
  最后一个 Pass CodegenPreproc和前置 Pass 是否通过？
       │
       ├── FAIL ──→ pass_compare 结果存在 Fail Op ──→ 定位到问题 Op，针对该 Op 检查实现 （CodegenPreproc 通过，前置 Pass 报错记录为精度工具框架问题，上报）
       │
       │ PASS
       │
       ▼
  特定问题排查
       │
       ├── 通过开关配置定位到 VF 融合 / Mix 子图 / 同步 / 合轴问题 ──→ 结束
       │
       └── 仍未定位 ──→ 进入 precision-binary-search 上板二分定位
```

---

## 一、Pass 校验

### 环境与配置

| 环境变量                    | 设置时机               | 说明             |
| --------------------------- | ---------------------- | ---------------- |
| `ASCEND_WORK_PATH`        | 运行测试前必须设置     | 组件日志输出目录 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为 0（DEBUG）    | 获取详细调试信息 |
| `TILE_FWK_DEVICE_ID`      | NPU 模式运行前必须设置 | 指定 NPU 设备 ID |

```bash
export ASCEND_WORK_PATH="/path/to/work/directory"
export ASCEND_GLOBAL_LOG_LEVEL=0
export TILE_FWK_DEVICE_ID=0
```

### 配置校验开关

#### verify_options 配置

在 PyPTO 算子实现文件中配置：

```python
verify_options = {
    "enable_pass_verify": True,            # 启用Pass验证（必须）
    "pass_verify_save_tensor": True,       # 保存中间数据
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    output: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1
```

| 配置项                      | 说明                                  | 默认值         |
| --------------------------- | ------------------------------------- | -------------- |
| `enable_pass_verify`      | 启用 Pass 验证                        | `False`      |
| `pass_verify_save_tensor` | 保存 Pass 中间数据                    | `False`      |
| `pass_verify_pass_filter` | 过滤要验证的 Pass（可选，见下方说明） | 默认 7 个 Pass |

> **`pass_verify_pass_filter` 取值说明**：
>
> - **不传**：默认校验 7 个 Pass：`ExpandFunction`、`ProcessAtomic`、`L1CopyInReuseMerge`、`InferDynShape`、`PreGraphProcess`、`InferParamIndex`、`CodegenPreproc`
> - **`["all"]`**：校验所有 Pass
> - **`[]`（空列表）**：不校验任何 Pass，只校验 tensor_graph

> **注意**：Shape 必须具体，不能使用占位符；返回类型必须使用输出参数形式。

#### tile_fwk_config.json 配置

配置文件位置：`framework/src/interface/configs/tile_fwk_config.json`

```json
{
    "global": {
        "pass": {
            "default_pass_configs": {
                "print_graph": true,
                "dump_graph": true,
                "pre_check": true,
                "post_check": true
            }
        }
    }
}
```

| 配置项          | 说明            | 使用场景           |
| --------------- | --------------- | ------------------ |
| `pre_check`   | Pass 执行前校验 | Pass精度问题时开启 |
| `post_check`  | Pass 执行后校验 | Pass精度问题时开启 |
| `dump_graph`  | 保存 IR 图      | 分析 Pass 处理结果 |
| `print_graph` | 打印 IR 图      | 快速查看图结构变化 |

> 大数据量时（Shape T>1000）：缩小shape参数、删除LOOP的unrolllist参数、增大tileshape（cube_tile_shapes、vec_tile_shapes）。

### 编译运行

```bash
python3 -m pip install . --verbose
python3 your_test_case.py
```

> **注意**: 所有需要重新编译的排查步骤（PreCheck/PostCheck、同步/VF等），均应使用当前 pypto 源码仓重新编译，不应因"需重新编译"而跳过。

输出目录：`./output/output_*`（验证数据）、`$ASCEND_WORK_PATH/log/`（日志）

### 分析验证结果

> **结论必须基于精度工具的实际输出**（`interpreter.log`、`pass_compare.py` 结果）。禁止仅凭代码注释、变量命名或推测得出根因结论。报告中 PASS / FAIL 判定必须引用 `interpreter.log` 中的具体行；根因定位必须引用 `pass_compare.py` 的输出数据。

错误码定义：`framework/include/tilefwk/error_code.h`

| 错误码       | 名称                   | 阶段      | 处理方法                                           |
| ------------ | ---------------------- | --------- | -------------------------------------------------- |
| `0xB4001U` | VERIFY_RESULT_MISMATCH | 前端/Pass | 参考[FAIL 处理](#fail-处理pass_compare-定位问题-op) |
| `0xB200FU` | RUNTIME_EXCEPTION      | Pass      | 检查 OP 属性，参考 IR 图                           |
| `0xB0001U` | VERIFY_NOT_ENABLE      | 环境      | 检查`torch >= 2.1.0`                             |
| 其他         | —                     | 未知      | 联系开发人员                                       |

> **⚠️ 判断标准（关键）**：**只看 CodegenPreproc Pass（最后一个 Pass）是否通过。**
>
> - **CodegenPreproc PASS** → 算子精度校验通过。前面任意 Pass（如 ReplaceTensor、ProcessAtomic、InferDynShape 等）出现 `VERIFY_RESULT_MISMATCH` 不影响算子精度判断，但**这些报错需要记录并上报**——它们属于**精度工具框架本身的误报**，应归类为"精度工具框架问题"，不作为算子精度 bug 追踪。
> - **CodegenPreproc FAIL** → 存在真实的精度差异，需要用 `pass_compare.py` 进一步定位。
>
> 如果遇到前述 Pass 报错但 CodegenPreproc 通过的情况，在报告中归类为**"精度工具框架问题（误报）"**，记录报错 Pass 和错误码并上报，不作为算子精度 bug 追踪。

### FAIL 处理：pass_compare 定位问题 Op

根据日志确定失败 Pass 后，使用 `pass_compare.py` 逐 Op 对比：

```bash
python3 tools/verifier/pass_compare.py \
    --p <FailedPass> <GoldenPass> \
    --verify_path=/path/to/verify_data
```

| 参数              | 必选 | 说明                                                                                   |
| ----------------- | ---- | -------------------------------------------------------------------------------------- |
| `--p`           | 是   | 两个 Pass 名称，空格分隔。第一个是问题 Pass，第二个是 golden Pass（前一个通过的 Pass） |
| `--verify_path` | 是   | verify 数据目录。可传1个（两个 Pass 在同一目录）或2个（分别对应两个 Pass）             |
| `--func`        | 否   | 指定对比的函数名，可传多个，默认对比所有函数                                           |
| `--atol`        | 否   | 绝对容差，默认 1e-3                                                                    |
| `--rtol`        | 否   | 相对容差，默认 1e-3                                                                    |
| `--topk`        | 否   | 失败时打印前 k 个差异元素，默认 1000                                                   |

> **定位结果**：pass_compare.py 生成 `verify_graph_result_cmp~Pass_xx~PassA~Pass_yy_PassB~timestamp.csv`，逐 Op 记录对比结果（PASS/FAIL/Skip），失败 Op 即为问题 Op。

**自动分析脚本**：

```bash
python3 -c "
import pandas as pd
cmp = pd.read_csv('<生成的CSV文件路径>')
total = len(cmp)
passed = sum(str(r).strip() == 'True' for r in cmp['AB>RESULT'])
failed = sum(str(r).strip() == 'False' for r in cmp['AB>RESULT'])
skipped = sum(str(r).strip() == 'Skip' for r in cmp['AB>RESULT'])
print(f'Total: {total}, Pass: {passed} ({passed/total*100:.1f}%), Fail: {failed}, Skip: {skipped}')
print()
if failed > 0:
    fails = cmp[cmp['AB>RESULT'].astype(str).str.strip() == 'False']
    print('=== Fail 按 opcode 分布 ===')
    for opcode, cnt in fails['B>:opcode'].value_counts().items():
        sub = fails[fails['B>:opcode'] == opcode]
        max_mae = sub['AB>mae'].astype(float).max()
        print(f'  {opcode:25s}  {cnt:4d}次  maxMAE={max_mae:.6f}')
    print()
    print('=== Fail 按 symbol 分布（最可能的问题 Op）===')
    print(fails[':symbol'].value_counts().to_string())
"
```

**分析结果判断**：

| 结果                     | 含义                        | 后续动作                                       |
| ------------------------ | --------------------------- | ---------------------------------------------- |
| 存在 Fail Op             | Fail 的 Op 即为精度问题来源 | 针对该 Op 检查实现逻辑、数据类型、shape 处理等 |
| 全部 Pass 但精度仍有问题 | Pass 层未检出差异           | 进入[二、特定问题排查](#二特定问题排查)         |

### 移除校验配置

Pass 校验完成后，**移除校验配置**，避免影响后续调试：

```python
@pypto.frontend.jit()  # 移除 verify_options 参数
def your_kernel(...)
```

恢复 `tile_fwk_config.json` 文件中 pre_check、post_check 配置。

---

## 二、特定问题排查

如果所有 Pass 校验都通过但算子精度仍有问题，需要排查以下特定场景。

> **⚠️ 分析边界**：以下排查仅通过开关配置（enable/disable）定界问题归属（同步/VF/合轴/Mix 合图），**禁止进一步分析框架内部根因**。不要阅读框架内部实现来推断根因。问题归属确认后即结束，反馈结果。
>
> 对特定问题的输出上限是："问题归属 XX，建议关闭/开启 XX 开关"。

**前置步骤**：确认设备型号（后续排查中部分场景仅针对特定设备）：

```bash
lspci | grep -i "acc" | grep -oE "d80[236]"
```

| 设备 ID  | 型号             |
| -------- | ---------------- |
| `d802` | Ascend 910B (A2) |
| `d803` | Ascend 910C (A3) |
| `d806` | Ascend 950 (A5)  |

### 同步/VF 融合问题

pipeline 同步不及时导致的数据竞争或 VF 融合问题会导致精度异常。

**快速验证**：

1. 修改 `framework/src/passes/block_graph_pass/insert_sync.h`，将 `bool enableDebug_{false}` 改为 `bool enableDebug_{true}`
2. 重新编译安装 pypto：`python3 -m pip install . --verbose`
3. 重新执行算子。若精度通过 → 说明是同步/VF 融合问题，恢复 `bool enableDebug_{false}` 参数，结束定位流程，反馈结果
4. 检查是否为 VF 融合问题（仅针对 A5 时执行，否则跳过，默认为同步问题）：
   - 修改 `framework/src/interface/configs/tile_fwk_config.json`，将 `"enable_vf": true` 改为 `"enable_vf": false`
   - 重新编译安装 pypto：`python3 -m pip install . --verbose`
   - 重新执行算子。若精度通过 → 说明是 VF 融合问题

若确认为同步问题导致精度失败，详细步骤请参考：**[references/pipe_all.md](references/pipe_all.md)**

**定位流程概览**：

1. 二分定位问题 CCE 文件（使用 `binary_pipeall_sync.py`）
2. 在问题 CCE 文件内手动二分插入 `pipe_barrier(PIPE_ALL)`，定位具体问题行
3. 使用 `locate_source_line.py` 映射问题行到前端源代码
4. 分析并修复同步问题

### 合轴问题

如果算子实现中开启了合轴（`pypto.experimental.set_operation_options(combine_axis=True)`），可以尝试关闭合轴，观察精度问题是否消失：

```python
pypto.experimental.set_operation_options(combine_axis=False)
```

若关闭后重新执行精度通过 → 说明是合轴引入的精度问题，结束定位流程，反馈结果。

### Mix 合图问题（仅 A5）

仅针对 A5（设备 ID `d806`）。如果算子实现中开启了自动 CV Mix 合图（`pypto.experimental.auto_mix_partition(1)`），可以尝试关闭合图，观察精度问题是否消失：

```python
pypto.experimental.auto_mix_partition(0)
```

若关闭后重新执行精度通过 → 说明是 Mix 合图引入的精度问题，结束定位流程，反馈结果。

---

## 打印上板信息

**前置条件**：已定位到具体 Op。

用于：打印上板tensor数据、验证动态shape/offset值、定位AICORE执行异常。

详细排查方法请参考：**[machine.md](../../../../docs/zh/guide/appendix/trouble_shooting/machine.md)**

### 打印环境配置

```json
{
    "global": {
        "codegen": {
            "fixed_output_path": true,
            "force_overwrite": false,
            "parallel_compile": 1
        }
    }
}
```

| 配置项                | 正确值    | 说明                              |
| --------------------- | --------- | --------------------------------- |
| `fixed_output_path` | `true`  | CCE固定生成在`./kernel_aicore/` |
| `force_overwrite`   | `false` | 不覆盖手动修改的CCE文件           |
| `parallel_compile`  | `1`     | 单线程编译，便于调试              |

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：

```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

### 可打印内容

| 内容          | 方法                    | 说明             |
| ------------- | ----------------------- | ---------------- |
| GM tensor数据 | `AiCorePrintGmTensor` | DDR/GM上的tensor |
| UB tensor数据 | `AiCorePrintUbTensor` | UB上的tensor     |
| Shape变量值   | `AICORE_LOGD`          | 动态shape实际值  |
| Offset值      | `AICORE_LOGD`          | 动态offset实际值 |

---

## 注意事项

1. **Pass 精度判断**：只看 CodegenPreproc。前置 Pass 报错不影响算子精度结论，但需记录为框架问题。
2. **框架误报上报**：CodegenPreproc PASS + 前置 Pass 报错 → 归类为"精度工具框架问题"，记录报错 Pass 名称和错误码，上报。
3. 动态shape验证/AICORE异常排查：参考 [machine.md](../../../../docs/zh/guide/appendix/trouble_shooting/machine.md)
4. 打印配置：`fixed_output_path=true`, `force_overwrite=false`
5. 打印限制：元素数量 ≤ 80
6. 配置备份：修改配置前建议备份原文件
7. 校验完成后移除配置：移除 `verify_options` 参数和 `tile_fwk_config.json` 中的校验开关，重新编译安装

---

## 相关文档

| 文档                                                                            | 内容                        |
| ------------------------------------------------------------------------------- | --------------------------- |
| [machine.md](../../../../docs/zh/guide/appendix/trouble_shooting/machine.md) | MACHINE组件错误码与排查指南 |
