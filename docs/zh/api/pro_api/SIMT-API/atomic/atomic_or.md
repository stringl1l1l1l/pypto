# pypto_pro.language.simt.atomic_or

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

以原子方式对目的操作数target与源操作数value执行按位或，并将结果写回目的操作数。

## 函数原型

```python
pypto_pro.language.simt.atomic_or(
    target: Scalar,
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | 目的操作数，Scalar类型。必须直接传入Tile或Tensor的单元素下标访问表达式，例如ub_tile[0, 0]或gm_tensor[0, 0]。<br>- UB Tile：必须位于UB，使用ND，支持DT_INT32、DT_UINT32。<br>- GM Tensor：必须为ND，支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64。 |
| value | 输入 | 源操作数，Scalar类型，表示按位或掩码。数据类型必须与target一致；整数字面量按target的数据类型处理。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回更新前的target值，返回值类型与target一致。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def merge_observed_flags(
    observed_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    flags: pl.Tensor[[1, 1], pl.DT_UINT32],
) -> None:
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_or(flags[0, 0], observed_flags[0, tid])


@pl.jit()
def atomic_or_kernel(
    observed_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    flags: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        merge_observed_flags[256](observed_flags, flags)
```
