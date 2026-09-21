# pypto.Tensor.set\_cache\_policy

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

设置Tensor Cache的属性。

## 函数原型

```python
set_cache_policy(self, policy: CachePolicy, value: bool) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| policy  | 输入      | 缓存策略类型,可选值为：<br> - CachePolicy.NONE_CACHEABLE：芯片提供了L2 Cache能力，但是有些算子的特点导致有L2 Cache反而不如没有L2 Cache，配置后此tensor不会经过L2 Cache，例如有以下常用场景：<br>   - 类似于weight这种常量，如果算子仅访问从out读取一次，不复用，则没有必要进L2；<br>   - 输出shape过大，下层算子最先使用的内存不是上层算子最后的输出结果，进L2后反而触发下层算子回写out，导致性能恶化。 |
| value   | 输入      | 是否启用该缓存策略。 |

## 返回值说明

无

## 约束说明

无。

## 调用示例

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
t.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```
