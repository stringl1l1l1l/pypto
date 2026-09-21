# PYPTO_PROF_PMU_EVENT_TYPE

## 功能描述

选择PMU（Performance Monitoring Unit）数据采集模式。PMU用于监控AI Core的硬件性能事件，不同模式对应不同的事件组。

- 类型：整数
- 取值范围：`1`、`2`、`4`、`5`、`6`、`7`、`8`。默认为`2`

## 配置示例

```bash
export PYPTO_PROF_PMU_EVENT_TYPE=2

# 采集PMU数据
msprof --task-time=l3 --output=./prof_data python xxx.py

# 解析数据
python tools/profiling/tilefwk_pmu_to_csv.py -p PROF_xxx/device_x/data -pe=$PYPTO_PROF_PMU_EVENT_TYPE --arch dav_3510
```

## 使用约束

- 完整的PMU采集流程详见[采集PMU数据](../../programming_guide/tensor/debug/performance.md)。

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
