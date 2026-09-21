# pypto.Element.dtype

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

获取数据类型。

## 函数原型

```python
dtype(self) -> pypto.DataType
```

## 参数说明

NA

## 返回值说明

返回数据类型。

## 约束说明

只读数据。

## 调用示例

```python
t = pypto.Element(pypto.DT_FP32, 3)
t.dtype
```
