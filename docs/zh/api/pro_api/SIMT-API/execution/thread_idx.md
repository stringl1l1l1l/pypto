# pypto_pro.language.simt.thread_idx

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

获取当前线程在SIMT线程块内的三维坐标。

线程是SIMT执行的最小编程单元。thread_idx()返回线程在线程块内的局部坐标，每个线程块都从零开始编号，因此不同线程块中的线程可以具有相同的线程坐标。

## 函数原型

```python
pypto_pro.language.simt.thread_idx() -> Any
```

## 参数说明

无。

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回三维线程坐标对象thread，通过thread.x、thread.y和thread.z读取各维坐标，每个分量均为DT_UINT32类型的Scalar。三个分量的取值范围依次为[0, block_dim().x)、[0, block_dim().y)和[0, block_dim().z)。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_thread_idx_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    thread = pl.simt.thread_idx()
    output[0, tid] = thread.x


@pl.jit()
def simt_thread_idx_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_thread_idx_x[256](output)
```
