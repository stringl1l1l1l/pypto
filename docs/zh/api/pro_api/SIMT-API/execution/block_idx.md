# pypto_pro.language.simt.block_idx

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

获取当前SIMT线程块在线程块网格中的三维坐标。

线程块坐标用于区分线程块网格中的不同线程块。同一线程块内的所有线程读取到相同的block_idx()，不同线程块通过不同的坐标处理各自的数据。

## 函数原型

```python
pypto_pro.language.simt.block_idx() -> Any
```

## 参数说明

无。

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回三维线程块坐标对象，通过block.x、block.y和block.z读取各维坐标，每个分量均为DT_UINT32类型的Scalar。当前线程块网格仅使用X维，block.x的范围为[0, grid_dim().x)，block.y和block.z均为0。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_block_idx_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    block = pl.simt.block_idx()
    output[0, tid] = block.x


@pl.jit()
def simt_block_idx_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_block_idx_x[256](output)
```
