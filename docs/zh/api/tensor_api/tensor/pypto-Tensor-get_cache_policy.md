# pypto.Tensor.get\_cache\_policy

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

获取某个缓存策略是否被启用。

## 函数原型

```python
get_cache_policy(self, policy: CachePolicy) -> bool
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| policy  | 输入      | 缓存策略类型。 |

## 返回值说明

Cache策略是否被启用。

## 约束说明

无。

## 调用示例

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
out = t.get_cache_policy(pypto.CachePolicy.PREFETCH)
```

结果示例如下：

```python
输出数据out: False
```
