# PYPTO_DATADUMP_ENABLE

## 功能描述

使能上板执行时的tensor dump功能。开启后，框架在NPU上执行算子时会自动dump leaf function的输入输出数据，用于与模拟计算结果进行精度对比分析。

- 类型：字符串
- 取值范围：`true`（开启）、`false`或未设置（关闭）

## 配置示例

```bash
# 方式一：命令行设置
export PYPTO_DATADUMP_ENABLE=true

# 方式二：Python代码中设置
import os
os.environ["PYPTO_DATADUMP_ENABLE"] = "true"
```

## 使用约束

- 需配合`verify_options`中的`enable_pass_verify`和`pass_verify_save_tensor`使用。
- dump数据落盘在`output/output_*/tensor/`目录下。
- 仅在NPU上板执行时生效，仿真模式下无效。

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
