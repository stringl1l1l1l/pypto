# pypto_pro.language.simt.isinf

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

判断源操作数是否为正无穷或负无穷。

## 函数原型

```python
pypto_pro.language.simt.isinf(
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

源操作数为正无穷或负无穷时返回True，否则返回False。返回值为DT_BOOL类型的Scalar。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def exp_or_fallback(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    result = pl.simt.exp(source[0, tid])
    output[0, tid] = result
    if pl.simt.isinf(result):
        output[0, tid] = fallback


@pl.jit()
def simt_isinf_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    with pl.section_vector():
        exp_or_fallback[256](source, output, fallback)
```
