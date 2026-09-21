# pypto.SymbolicScalar构造函数

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

创建新的SymbolicScalar实例，支持多种构造方式。

## 函数原型

```python
__init__(self,
         arg0: Union[int, str, 'SymbolicScalar'] = None,
         arg1: Union[int, None] = None
) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self    | 输入      | 实例对象引用，Python自动传递。 |
| arg0    | 输入      | 符号标量的值或名称，可以是：<br> - int：整数值，创建一个常量符号标量<br> - str：符号名称，创建一个符号标量<br> - SymbolicScalar：另一个符号标量，用于复制 |
| arg1    | 输入      | 符号标量的值，仅在arg0为字符串时可选择使用。 |

## 返回值说明

无。

## 约束说明

- 如果arg0为None，将创建一个无初始值的符号标量
- 如果arg0是整数，arg1将被忽略
- 如果arg0是字符串且arg1是整数，将创建一个带初始值的符号标量
- 如果arg0是字符串且arg1是None，将创建一个无初始值的符号标量
- 如果arg0是SymbolicScalar，将复制其底层实现对象

## 调用示例

```python
a = pypto.SymbolicScalar()
b = pypto.SymbolicScalar(10)
c = pypto.SymbolicScalar("x")
d = pypto.SymbolicScalar("x", 10)
e = pypto.SymbolicScalar(c)
```
