# pypto_pro.language.simt.grid_dim

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

获取当前线程块网格在X、Y、Z三个维度上的线程块数量。

线程块网格（Grid）的尺寸来自外层Kernel实际生效的Vector执行域，可能受运行时限核影响。网格中的线程块总数为X、Y、Z三个维度大小的乘积。

## 函数原型

```python
pypto_pro.language.simt.grid_dim() -> Any
```

## 参数说明

无。

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回三维线程块网格尺寸对象，通过grid.x、grid.y和grid.z读取各维大小，每个分量均为DT_UINT32类型的Scalar。当前线程块网格仅使用X维，grid.x为外层Kernel的Vector执行域逻辑Block数量，grid.y和grid.z均为1。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_grid_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    grid = pl.simt.grid_dim()
    output[0, tid] = grid.x


@pl.jit()
def simt_grid_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_grid_dim_x[256](output)
```
