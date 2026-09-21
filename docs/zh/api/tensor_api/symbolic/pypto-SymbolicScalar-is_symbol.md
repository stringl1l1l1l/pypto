# pypto.SymbolicScalar.is\_symbol

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

判断符号标量是否为符号。

## 函数原型

```python
is_symbol(self) -> bool
```

## 参数说明

无

## 返回值说明

如果是符号返回True，不是返回False。

## 约束说明

符号变量是指在运行时无法确定具体数值、用于构建计算图的变量

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_symbol()
out2 = s2.is_symbol()
```

结果示例如下：

```python
输出数据out1: False
输出数据out2: True
```
