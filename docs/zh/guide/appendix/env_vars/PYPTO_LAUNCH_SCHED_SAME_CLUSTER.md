# PYPTO_LAUNCH_SCHED_SAME_CLUSTER

## 功能描述

控制是否强制在同一Cluster内分配调度线程（AICPU）。在同一Cluster分配调度线程能够获得更好的核间流水性能，但在整网场景中，除PyPTO外还有其他组件使用AICPU，强制同Cluster可能因AICPU资源不足导致执行超时。

- 类型：字符串
- 取值范围：`true`（默认，同Cluster分配）、`false`（不强制同Cluster）

## 配置示例

```bash
# 整网场景下关闭同Cluster约束，避免AICPU资源不足
export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false
```

## 使用约束

- 设置为`false`时，可以配合`launch_sched_aicpu_num`（通过`runtime_options`配置）指定可用的AICPU数量。
- 开启同Cluster分配时，`launch_sched_aicpu_num`配置不生效。

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
