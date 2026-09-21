# pypto.set\_codegen\_options

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

设置codegen的选项。

## 函数原型

```python
set_codegen_options(*,
                    support_dynamic_aligned: bool = None,
                    enable_pmu_trace: bool = None) -> None
```

## 参数说明

| 参数名                      | 输入/输出 | 说明                                                                 |
|-----------------------------|-----------|----------------------------------------------------------------------|
| support_dynamic_aligned     | 输入      | 含义：是否支持动态Shape。 <br> 说明： <br> 当值为True，算子生成的设备侧二进制可支持动态Shape对齐场景。 <br> 当值为False，算子生成的设备侧二进制仅支持处理动态Shape非对齐场景。 <br> 类型：bool <br> 取值范围：{True, False} <br> 默认值：False（当算子确认动态Shape，且Shape尾轴均为对齐时，可尝试打开确认是否有性能收益） <br> 影响Pass范围：无，仅影响CodeGen模块生成设备侧目标代码 |
| enable_pmu_trace            | 输入      | 含义：是否开启 PMU Trace。 <br> 说明： <br> 当值为True，codegen 会在每个 leaf function 对应 CCE 代码的首尾插入 `bisheng::cce::mark_stamp` 打点，并为其分配 PMU ID，用于在核内流水中定位对应 kernel；运行时需配合 `msprof --instr-profiling=on` 进行指令级采集。 <br> 当值为False，不插入 PMU 打点。 <br> 类型：bool <br> 取值范围：{True, False} <br> 默认值：False <br> 影响Pass范围：无，仅影响 CodeGen 模块生成设备侧目标代码 |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

support\_dynamic\_aligned选项效果后续会通过Pass推导机制进行优化，无需用户手工设置，建议用户谨慎使用。

enable\_pmu\_trace使用说明：
<!-- npu="950" id4 -->
- Ascend 950PR&950DT系列产品：仅支持单算子采集（不支持整网场景），且最多只能采集 6 个核的数据。更多使用说明参见[性能调优指南](../../../guide/programming_guide/tensor/debug/performance.md#pmu-trace)。
<!-- end id4 -->
<!-- npu="A3" id5 -->
- Atlas A3系列产品：不支持。
<!-- end id5 -->
<!-- npu="910b" id6 -->
- Atlas A2系列产品：不支持
<!-- end id6 -->

## 调用示例

```python
pypto.set_codegen_options(support_dynamic_aligned=True)
pypto.set_codegen_options(enable_pmu_trace=True)
```
