# pypto_pro.language.make_ptr

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

从已有Tensor获取地址，或为已有Ptr创建新的元素类型视图。返回的Ptr与源对象共享地址；指定dtype时，返回的Ptr按该元素类型解释数据。

常用于将按字节粒度申请的workspace按更宽的数据类型解释，再配合[pypto_pro.language.make_tensor](make_tensor.md)包装成可load或store的Tensor视图。addptr按元素偏移指针起点并保留元素类型；make_ptr保留指针起点并设置元素类型。

## 函数原型

```python
pypto_pro.language.make_ptr(ptr: Union[Tensor, Ptr], dtype: Optional[DataType] = None) -> Ptr
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| ptr | 输入 | 待转换的对象，Tensor或Ptr类型。传入Tensor时获取其地址；传入Ptr时复用原地址。本接口不分配内存，也不复制数据。 |
| dtype | 输入 | 目标元素类型，[DataType](../basic_data_structures/DataType.md)类型，可选。传入时，返回的Ptr按该数据类型解释原数据；省略时保留源对象的元素类型。指定dtype不会改变原数据，调用方必须保证新类型满足地址对齐和内存范围要求。 |

## 约束说明

无。

## 返回值说明

返回与源Tensor或源指针地址相同、元素类型为dtype的Ptr；省略dtype时保留源对象的元素类型。

## 调用示例

### 将DT_UINT8指针重解释为DT_FP16指针

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def make_ptr_kernel(
    workspace: pl.Ptr[pl.DT_UINT8],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    fp16_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP16)
    ws_buf = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```

### 保留源指针的数据类型

```python
same = pl.make_ptr(ptr)
```

### 从Tensor提取裸指针

```python
ptr = pl.make_ptr(tensor)
```
