# pypto_pro.language.simt.atomic_add

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

以原子方式将源操作数value累加到目的操作数target中。

## 函数原型

```python
pypto_pro.language.simt.atomic_add(
    target: Scalar,
    value: Scalar,
) -> Optional[Scalar]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | 目的操作数，Scalar类型。必须直接传入Tile或Tensor的单元素下标访问表达式，例如ub_tile[0, 0]或gm_tensor[0, 0]。<br>- UB Tile：必须位于UB，使用ND，支持DT_INT32、DT_UINT32、DT_FP16、DT_BF16、DT_FP32。<br>- GM Tensor：必须为ND，支持DT_INT32、DT_UINT32、DT_FP16、DT_BF16、DT_FP32、DT_INT64、DT_UINT64。 |
| value | 输入 | 源操作数，Scalar类型，表示累加值。数据类型必须与target一致；数值字面量按target的数据类型处理，整数目的操作数不接受浮点字面量。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

- target的数据类型为DT_FP16或DT_BF16时无返回值，只能作为独立语句调用；
- target的数据类型为支持的其他数据类型时，返回更新前的target值，返回值类型与target一致。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def atomic_add_histogram_gm(
    histogram: pl.Tensor[[1, 16], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_add(histogram[0, tid % 16], 1)


@pl.jit()
def simt_atomic_add_histogram_gm(
    histogram: pl.Tensor[[1, 16], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_add_histogram_gm[256](histogram)
```
