# pypto.SymbolicScalar.as\_variable

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

将符号标量标记为中间变量。

## 函数原型

```python
as_variable(self) -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 这是一个原地操作，会修改符号标量的内部状态
- 通常用于优化表达式，将复杂表达式标记为中间变量

## 调用示例

```python
s = pypto.SymbolicScalar("x")
s.as_variable()  # 标记为中间变量
```
