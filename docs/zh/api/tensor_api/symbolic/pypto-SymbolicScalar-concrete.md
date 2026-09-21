# pypto.SymbolicScalar.concrete

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

获取符号标量的具体数值。

## 函数原型

```python
concrete(self) -> int
```

## 参数说明

无

## 返回值说明

符号标量的具体数值。

## 约束说明

- 只有在is\_concrete\(\)返回True时才能调用此方法
- 如果符号标量不是具体的，将抛出ValueError异常

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# 如果符号标量没有具体值会抛出异常
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

结果示例如下：

```python
输出数据out1: 10
输出数据out2: 抛出异常ValueError: Not concrete value
```
