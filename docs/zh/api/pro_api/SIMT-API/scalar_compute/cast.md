# pypto_pro.language.simt.cast

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

将源操作数转换为指定的数据类型。转换过程中可通过mode参数指定舍入模式。

## 函数原型

```python
pypto_pro.language.simt.cast(
    value: Scalar,
    dtype: DType,
    *,
    mode: RoundMode = pl.RoundMode.CAST_NONE,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_FP16、DT_BF16、DT_FP32、DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |
| dtype | 输入 | 目的数据类型，DType类型，支持的数据类型转换见下表。 |
| mode | 输入 | 可选，舍入模式，[RoundMode](../../SIMD-API/basic_data_structures/RoundMode.md)类型，默认值为pypto_pro.language.RoundMode.CAST_NONE。各数据类型转换支持的舍入模式见下表。 |

各源数据类型支持的目的数据类型和舍入模式如下：

<table>
  <thead>
    <tr>
      <th>源数据类型</th>
      <th>目的数据类型</th>
      <th>支持的mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">DT_FP16</td>
      <td>DT_FP16、DT_FP32</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_BF16、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_INT16、DT_UINT16</td>
      <td>CAST_RINT、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_INT8、DT_UINT8</td>
      <td>CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="4">DT_BF16</td>
      <td>DT_BF16、DT_FP32</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_INT16、DT_UINT16</td>
      <td>CAST_RINT、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_INT8、DT_UINT8</td>
      <td>CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="3">DT_FP32</td>
      <td>DT_FP32</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_ODD</td>
    </tr>
    <tr>
      <td>DT_BF16、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_INT8</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td rowspan="2">DT_INT16</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="3">DT_INT32</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_FP32</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="3">DT_INT64</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_FP32</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_UINT8</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td rowspan="2">DT_UINT16</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="3">DT_UINT32</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_FP32</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td rowspan="3">DT_UINT64</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64</td>
      <td>CAST_NONE</td>
    </tr>
    <tr>
      <td>DT_FP16、DT_BF16</td>
      <td>CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
    <tr>
      <td>DT_FP32</td>
      <td>CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

- DT_FP16或DT_BF16转换为DT_INT8、DT_UINT8、DT_INT16、DT_UINT16时，舍入后的结果会钳位到目的整数类型的取值范围。

## 返回值说明

返回转换后的Scalar，数据类型由dtype指定。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def cast_floor_fp32_to_int32(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cast(
        source[0, tid],
        pl.DT_INT32,
        mode=pl.RoundMode.CAST_FLOOR,
    )


@pl.jit()
def simt_cast_floor_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    with pl.section_vector():
        cast_floor_fp32_to_int32[256](source, output)
```
