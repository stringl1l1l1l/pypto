# pypto.SymbolicScalar.is\_concrete

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

判断符号标量是否有具体的数值。

## 函数原型

```python
is_concrete(self) -> bool
```

## 参数说明

无

## 返回值说明

如果有具体数值返回True，否则返回False。

## 约束说明

- 常量值总是具体的
- 某些表达式在特定条件下也可能具有具体值

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.is_concrete()
s2 = pypto.SymbolicScalar("x")
out2 = s2.is_concrete()
```

结果示例如下：

```python
输出数据out1: True
输出数据out2: False
```
