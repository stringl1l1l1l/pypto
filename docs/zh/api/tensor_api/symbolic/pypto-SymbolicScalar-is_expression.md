# pypto.SymbolicScalar.is\_expression

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

判断符号标量是否为表达式。

## 函数原型

```python
is_expression(self) -> bool
```

## 参数说明

无

## 返回值说明

如果是表达式返回True，不是返回False。

## 约束说明

表达式是由多个符号或常量通过运算组合而成的。

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
s3 = s2 + 5
out1 = s1.is_expression()
out2 = s2.is_expression()
out3 = s3.is_expression()
```

结果示例如下：

```python
输出数据out1: False
输出数据out2: False
输出数据out3: True
```
