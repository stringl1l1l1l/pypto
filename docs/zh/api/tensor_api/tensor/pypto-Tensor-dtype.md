# pypto.Tensor.dtype

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

获取Tensor的数据类型。

## 函数原型

```python
dtype(self) -> DataType
```

## 参数说明

无

## 返回值说明

返回Tensor的数据类型。

## 约束说明

无。

## 调用示例

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
out = t.dtype
```

结果示例如下：

```python
输出数据out: DataType.DT_FP32
```
