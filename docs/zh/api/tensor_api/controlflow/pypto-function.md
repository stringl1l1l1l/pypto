# pypto.function

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

定义一个PyPTO计算函数。在该函数下可添加构建计算图所需要的操作。

## 函数原型

```python
function(name: str, *args, **kwargs) -> Iterator
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| name   | 输入      | 函数的名称，用于标识该计算图。 |
| *args  | 输入      | 用于接收传入的Tensor参数。 |

## 返回值说明

返回一个上下文管理器，在with语句中使用

## 约束说明

无。

## 调用示例

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```
