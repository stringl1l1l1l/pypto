# TILE_FWK_DEVICE_ID

## 功能描述

指定PyPTO算子执行时使用的NPU设备卡号。框架和测试用例通过该变量确定目标设备。

- 类型：整数
- 取值范围：`0` ~ `N-1`（N为可用NPU数量）

## 配置示例

```bash
export TILE_FWK_DEVICE_ID=0
```

## 使用约束

- 指定的设备卡号必须为可用状态，可通过`npu-smi info`确认。
- 多进程场景下应避免多个进程使用同一设备卡号。

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
