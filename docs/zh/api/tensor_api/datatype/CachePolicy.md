# CachePolicy

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

CachePolicy定义了Tensor的缓存策略，用于控制数据在各级缓存中的行为，优化内存访问性能和减少内存带宽消耗。

## 原型定义

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # 预取策略，提前将数据加载到缓存中，减少访问延迟
     NONE_CACHEABLE = ...  # 不可缓存策略，数据不存储在缓存中，直接访问主存
```
