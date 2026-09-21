# TopKAlgo

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

TopKAlgo定义了TopK的算法，用于控制TopK计算时的处理方式。

## 原型定义

```python
class TopKAlgo(enum.Enum):
     MERGE_SORT = ...    # 归并排序算法
     RADIX_SELECT = ...  # 基数选择算法
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| MERGE_SORT | 归并排序算法。对整个张量排序，之后选出前k个数。 |
| RADIX_SELECT | 基数选择算法。先找出第k个数，之后根据第k个数找出前k个数。 |

## 使用建议

1. **默认行为**：如果不指定算法，默认使用`MERGE_SORT`模式。
2. **性能要求高的场景**：推荐使用`RADIX_SELECT`模式，时间复杂度为O\(n\)。
3. **对性能要求不高的场景**：可以使用`MERGE_SORT`模式，时间复杂度为O\(nlogn\)。

## 使用示例

```python
import pypto

# 创建张量
x = pypto.tensor([2, 3], pypto.DT_FP32)

# 使用归并排序算法
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)

# 使用基数选择算法
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.RADIX_SELECT)

# 默认使用归并排序算法
y = pypto.topk(x, 2, -1, True)
```
