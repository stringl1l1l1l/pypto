# pypto_pro.language.trap

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

强制中止Kernel执行，用于调试。无条件中止，不接受任何参数。

## 函数原型

```python
pypto_pro.language.trap() -> None
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl
# 条件中止
if flag:
    pl.trap()

# 无条件中止（调试用）
pl.trap()
```
