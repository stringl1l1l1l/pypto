# pypto.Tensor.move

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

将一个Tensor数据移动到当前Tensor。

## 函数原型

```python
move(self, other: 'Tensor') -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 要移动数据的源Tensor。 |

## 返回值说明

无

## 约束说明

无。

## 调用示例

```python
t1 = pypto.tensor((2, 3), pypto.DT_FP32)
t2 = pypto.tensor((2, 3), pypto.DT_FP32)
# 将t2的数据移动到t1
t1.move(t2)
```
