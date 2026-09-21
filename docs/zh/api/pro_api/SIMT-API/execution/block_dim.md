# pypto_pro.language.simt.block_dim

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

获取当前SIMT线程块在X、Y、Z三个维度上的线程数。

线程块是SIMT函数的启动单位。块内线程执行同一份SIMT函数，并通过线程索引处理不同的数据。

## 函数原型

```python
pypto_pro.language.simt.block_dim() -> Any
```

## 参数说明

无。

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回三维线程块尺寸对象。线程块尺寸由外层JIT Kernel调用`simt_func[threads](...)`时通过方括号中的`threads`设置。通过dimensions.x、dimensions.y和dimensions.z读取各维大小，每个分量均为DT_UINT32类型的Scalar。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_block_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    dimensions = pl.simt.block_dim()
    output[0, tid] = dimensions.x


@pl.jit()
def simt_block_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_block_dim_x[256](output)
```
