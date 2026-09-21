# pypto_pro.language.simt.fma

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

执行融合乘加运算，在一次融合运算中完成乘法和加法，中间乘积不单独舍入。计算公式如下：

$$result = lhs \times rhs + addend$$

## 函数原型

```python
pypto_pro.language.simt.fma(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数（乘数），Scalar类型，支持DT_FP16、DT_BF16和DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数（乘数），Scalar类型，数据类型必须与lhs一致。Tensor或Tile元素需通过下标访问后传入。 |
| addend | 输入 | 源操作数（加数），Scalar类型，数据类型必须与lhs和rhs一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回融合乘加结果，数据类型与输入一致。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def linear(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fma(source[0, tid], scale, bias)


@pl.jit()
def simt_fma_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        linear[256](source, output, scale, bias)
```
