# pypto_pro.language.simt.sqrt

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

计算源操作数的主平方根，计算公式如下：

$$result = \sqrt{value}$$

## 函数原型

```python
pypto_pro.language.simt.sqrt(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_FP16、DT_BF16和DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回源操作数的主平方根，数据类型与输入一致。特殊值如下：

| value取值 | 返回值 |
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf、NaN或有限负数 | NaN |

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def vector_length(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    y: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    squared = x[0, tid] * x[0, tid] + y[0, tid] * y[0, tid]
    output[0, tid] = pl.simt.sqrt(squared)


@pl.jit()
def simt_sqrt_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    y: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        vector_length[256](x, y, output)
```
