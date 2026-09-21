# DUMP_DEVICE_PERF

## 功能描述

使能AICPU / AI Core联合性能数据采集。开启后，框架在执行过程中采集AICPU调度与AI Core执行过程中端到端的耗时数据，并能够在终端打屏，输出相关性能统计信息。采集结果落盘到`output/output_时间戳/`目录下。

- 类型：字符串
- 取值范围：`true`（开启）、`false`或未设置（关闭）

## 配置示例

```bash
export DUMP_DEVICE_PERF=true
python3 examples/02_intermediate/operators/softmax/softmax.py

# 采集完成后分析数据
python tools/scripts/machine_perf_trace.py analyze output/output_<时间戳>/machine_trace_perf_data_0.json
```

## 使用约束

- 最多支持200轮数据采集和打屏，超出部分将被截断。
- 当前只支持采集200次devTask构建的数据，超出时日志中会出现告警。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A3系列产品：支持
<!-- end id3 -->
