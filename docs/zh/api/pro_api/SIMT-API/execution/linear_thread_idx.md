# pypto_pro.language.simt.linear_thread_idx

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

获取当前线程在SIMT线程块内按X维优先展开的一维编号。

三维线程坐标按照X维优先的顺序展开，计算公式如下：

$$result = thread\_idx().x + thread\_idx().y \times block\_dim().x + thread\_idx().z \times block\_dim().x \times block\_dim().y$$

## 函数原型

```python
pypto_pro.language.simt.linear_thread_idx() -> Scalar
```

## 参数说明

无。

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回DT_UINT32类型的Scalar，表示当前线程在SIMT线程块内的一维编号，取值范围从0开始，最大值为当前线程块线程总数减1。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def copy_by_linear_thread_idx(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = source[0, tid]


@pl.jit()
def simt_linear_thread_idx_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        copy_by_linear_thread_idx[256](source, output)
```
