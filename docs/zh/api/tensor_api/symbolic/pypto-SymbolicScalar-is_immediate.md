# pypto.SymbolicScalar.is\_immediate

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

判断符号标量是否为立即值（具体数值）。

## 函数原型

```python
is_immediate(self) -> bool
```

## 参数说明

无

## 返回值说明

如果是立即值返回True，不是返回False。

## 约束说明

立即值是指在编译时就能确定具体数值的常量

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

结果示例如下：

```python
输出数据out1: True
输出数据out2: False
```
