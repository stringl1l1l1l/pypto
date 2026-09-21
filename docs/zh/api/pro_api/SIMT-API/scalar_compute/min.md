# pypto_pro.language.simt.min

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

比较两个源操作数并返回较小值，计算公式如下：

$$result = \min(lhs, rhs)$$

## 函数原型

```python
pypto_pro.language.simt.min(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_FP16、DT_BF16、DT_FP32、DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，数据类型必须与lhs一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回两个源操作数中的较小值，数据类型与输入一致。对于浮点输入，一个操作数为NaN时返回另一个操作数，两个操作数均为NaN时返回NaN。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def clamp_upper(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.min(source[0, tid], upper)


@pl.jit()
def simt_min_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    with pl.section_vector():
        clamp_upper[256](source, output, upper)
```
