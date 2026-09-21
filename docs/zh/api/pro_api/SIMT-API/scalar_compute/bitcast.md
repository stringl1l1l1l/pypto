# pypto_pro.language.simt.bitcast

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

将源操作数的二进制位模式重新解释为指定的等宽数据类型，不进行数值转换、舍入或饱和处理。

## 函数原型

```python
pypto_pro.language.simt.bitcast(
    value: Scalar,
    dtype: DType,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型。Tensor或Tile元素需通过下标访问后传入。 |
| dtype | 输入 | 目的数据类型，DType类型，支持的数据类型转换见下表。 |

各源数据类型支持的目的数据类型如下：

| 源数据类型 | 目的数据类型 |
|---|---|
| DT_FP16 | DT_INT16、DT_UINT16 |
| DT_BF16 | DT_INT16、DT_UINT16 |
| DT_FP32 | DT_INT32、DT_UINT32 |
| DT_INT16 | DT_FP16、DT_BF16 |
| DT_UINT16 | DT_FP16、DT_BF16 |
| DT_INT32 | DT_FP32 |
| DT_UINT32 | DT_FP32 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

- 源数据类型和目的数据类型的位宽必须相同，不支持相同数据类型之间的转换。

- NaN、无穷、负零和无效浮点编码均按原始位模式保留，不进行规范化。

## 返回值说明

返回与value具有相同二进制位模式、数据类型为dtype的Scalar。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=256)
def fp32_bits(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.bitcast(source[0, tid], pl.DT_UINT32)


@pl.jit()
def simt_fp32_bits_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        fp32_bits[256](source, output)
```
