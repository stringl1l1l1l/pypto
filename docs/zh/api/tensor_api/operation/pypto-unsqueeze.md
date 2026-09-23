# pypto.unsqueeze

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

为输入Tensor增加维度。

## 函数原型

```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：PyPTO支持的数据类型<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定插入新维度的位置（索引）。<br>支持负索引。<br>需在 [-input.dim - 1, input.dim] 范围内。 |

## 约束说明

输入Tensor不支持动态轴，即input的shape中的任何轴都不能标记为`pypto.DYNAMIC`。

## 返回值说明

返回在指定维度dim处新增大小为1的维度的输出Tensor，与输入Tensor共享数据且属性一致。

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.unsqueeze(x, 0)
```

结果示例如下：

```python
输出数据x: [[1, 2, 3],
            [4, 5, 6]]
输出数据y: [[[1, 2, 3],
             [4, 5, 6]]]
```
