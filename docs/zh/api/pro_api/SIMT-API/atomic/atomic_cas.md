# pypto_pro.language.simt.atomic_cas

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

以原子方式比较目的操作数target的旧值与源操作数compare。两者相等时将源操作数value写入目的操作数，否则保持目的操作数不变。

## 函数原型

```python
pypto_pro.language.simt.atomic_cas(
    target: Scalar,
    compare: Scalar,
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | 目的操作数，Scalar类型。必须直接传入Tile或Tensor的单元素下标访问表达式，例如ub_tile[0, 0]或gm_tensor[0, 0]。<br>- UB Tile：必须位于UB，使用ND，支持DT_INT32、DT_UINT32、DT_FP32。<br>- GM Tensor：必须为ND，支持DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| compare | 输入 | 源操作数，Scalar类型，表示期望值。数据类型必须与target一致；数值字面量按target的数据类型处理，整数目的操作数不接受浮点字面量。 |
| value | 输入 | 源操作数，Scalar类型，表示比较相等时写入的新值。数据类型必须与target一致；数值字面量按target的数据类型处理，整数目的操作数不接受浮点字面量。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回比较前的target值，返回值类型与target一致。可通过返回值是否等于compare判断本次交换是否成功。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=32)
def atomic_cas_winner_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 32], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_cas(state[0, 0], 0, 1)


@pl.jit()
def simt_atomic_cas_winner_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 32], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_cas_winner_gm[32](state, old_values)
```
