# pypto.Element构造函数

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

创建Element

## 函数原型

```python
__init__(self, dtype, value)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                  |
|--------|-----------|-----------------------|
| dtype  | 输入      | 数据类型，详见[DataType](../datatype/DataType.md)。|
| value  | 输入      | 整数或者浮点数。       |

## 返回值说明

返回Element

## 约束说明

无。

## 调用示例

```python
t = pypto.Element(pypto.DT_FP32, 3)
```
