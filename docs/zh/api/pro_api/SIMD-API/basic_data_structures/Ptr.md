# pypto_pro.language.Ptr

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

指向指定元素类型的GM地址类型标注。

pypto_pro.language.Ptr主要用于：

1. Kernel函数签名中声明GM裸指针参数
2. 配合[pypto_pro.language.make_ptr](../resource_management/make_ptr.md)创建不同元素类型的指针视图
3. 配合[pypto_pro.language.addptr](../resource_management/addptr.md)做指针偏移
4. 配合[pypto_pro.language.make_tensor](../resource_management/make_tensor.md)从裸指针构造Tensor view

## 函数原型

```python
pypto_pro.language.Ptr[dtype]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dtype | 输入 | 指针指向的元素数据类型，[pypto_pro.language.DataType](DataType.md)枚举值。常用取值：pypto_pro.language.DT_FP16、pypto_pro.language.DT_FP32、pypto_pro.language.DT_INT8。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

本示例在Kernel签名中使用pypto_pro.language.Ptr声明workspace指针，通过pypto_pro.language.addptr将指针偏移64 × 128个FP16元素，再使用pypto_pro.language.make_tensor创建形状为[64, 128]的Tensor视图。示例将输入Tensor中的数据搬入UB，逐元素乘2后写入workspace，再从workspace搬入UB并写入输出Tensor。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```
