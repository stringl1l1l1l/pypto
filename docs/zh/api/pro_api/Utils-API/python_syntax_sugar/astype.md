# pypto_pro.language.astype

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

将标量表达式的结果转换为指定数据类型。

## 函数原型

```python
result = pypto_pro.language.astype(x, dtype)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| x | 输入 | 源标量表达式，Scalar类型，按位置传递。 |
| dtype | 输入 | 目标数据类型，DataType类型，按位置传递。 |

## 约束说明

无。

## 返回值说明

返回数据类型为dtype的Scalar。

## 调用示例

```python
import pypto_pro.language as pl

# 将INT32标量转换为INT64
source = pl.const(1, pl.DT_INT32)
converted = pl.astype(source, pl.DT_INT64)

# 显式统一控制流两个分支的标量类型
if flag:
    value = pl.astype(lhs, pl.DT_INT64)
else:
    value = pl.astype(rhs, pl.DT_INT64)
```
