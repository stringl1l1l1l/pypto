# pypto_pro.language.addptr

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

对一个裸指针（pypto_pro.language.Ptr[dtype]）做偏移运算，得到指向同一片GM、起点平移后的新指针。偏移以**元素**为单位（不是字节），元素大小由指针的dtype决定。

常用于把一块workspace（GM暂存区）按需切成多段，配合[pypto_pro.language.make_tensor](make_tensor.md)把每段包装成可load/store的Tensor视图。

## 函数原型

```python
pypto_pro.language.addptr(ptr: Ptr, offset: Union[int, Scalar]) -> Ptr
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| ptr | 输入 | 输入指针，Ptr类型。返回的新指针与ptr的数据类型相同。元素位宽必须不少于8 bit；亚字节数据类型不支持指针算术，可先通过make_ptr重解释为DT_UINT8等字节可寻址类型。 |
| offset | 输入 | 指针偏移量，int或Scalar类型，单位为元素。编译器根据指针的数据类型换算实际字节偏移，偏移后的地址必须仍位于原workspace范围内。 |

## 约束说明

无。

## 返回值说明

返回与ptr数据类型相同、地址偏移后的Ptr。

## 调用示例

### 偏移workspace指针并创建Tensor视图

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
