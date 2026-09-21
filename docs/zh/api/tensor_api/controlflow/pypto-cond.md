# pypto.cond

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

定义一个if条件操作，实现python中的if功能。

## 函数原型

```python
cond(scalar: SymInt) -> pypto_impl.RecordIfBranch
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| scalar | 输入      | 条件表达式，可以是整数或SymbolicScalar（符号标量），用于判断条件是否为真 |

## 返回值说明

pypto\_impl.RecordIfBranch: 返回一个条件分支对象，用于Python的if语句

## 约束说明

- 必须与Python的if、elif、else语句配合使用
- 条件表达式会被记录到计算图中
- 支持嵌套的条件语句
- 当函数未使用 @pypto.frontend.jit或 @pypto.frontend.function装饰器修饰时，条件表达式需要用pypto.cond包装

## 调用示例

```python
# 未使用装饰器，需要用pypto.cond包装条件表达式
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_end(s2_idx, bn_per_batch)):
            ...

# 使用装饰器，无需pypto.cond包装
@pypto.frontend.jit
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.is_loop_end(s2_idx, bn_per_batch):
            ...
```
