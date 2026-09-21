# pypto.Element.value

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

获取数据。

## 函数原型

```python
value(self) -> int | float
```

## 参数说明

NA

## 返回值说明

返回Element中的数据。

## 约束说明

只读属性。

## 调用示例

```python
t = pypto.Element(pypto.DT_FP32, 3)
t.value
```
