# vf.create_addr_reg

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

vf.create_addr_reg用于创建地址偏移量寄存器（AddrReg），在多维循环中逐层累加地址偏移。AddrReg可作为vf.load_align和vf.store_align的地址偏移参数，替代直接传入整数偏移量。

偏移量由各循环轴对应的stride决定，支持1-4层循环轴。在循环中，AddrReg的偏移量按各循环轴对应的stride自动累加。

## 函数原型

```python
create_addr_reg(stride0, stride1=None, stride2=None, stride3=None, dtype: Optional[DType] = None) -> a_reg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| stride0 | 输入 | 最外层循环轴对应的地址偏移量，单位为元素个数。 |
| stride1 | 输入 | 可选，第二层循环轴对应的地址偏移量，单位为元素个数。 |
| stride2 | 输入 | 可选，第三层循环轴对应的地址偏移量，单位为元素个数。 |
| stride3 | 输入 | 可选，第四层循环轴对应的地址偏移量，单位为元素个数。 |
| dtype | 输入 | 可选，模板参数对应的数据类型（默认pypto_pro.language.DT_FP32）。决定元素宽度：8位宽（DT_INT8、DT_UINT8）/16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）/32位宽（DT_INT32、DT_UINT32、DT_FP32）/64位宽（DT_INT64、DT_UINT64）。 |

## 约束说明

- 数据类型约束：

  支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。

## 返回值说明

返回a_reg目的操作数，AddrReg地址偏移量寄存器。<br>- AddrReg数量上限为8。<br>- 由于硬件循环限制，AddrReg最多支持4层循环轴。<br>-AddrReg仅支持vf.load_align和vf.store_align搬运指令使用。<br>- 通过AddrReg设置地址偏移进行搬运时，需要满足对应搬运指令的地址对齐约束。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_repeat_size = 64
    repeat_times = 2
    for i in pl.range(0, repeat_times, 1):
        a_reg = vf.create_addr_reg(one_repeat_size, dtype=pl.DT_FP32)
        reg = vf.load_align(src_tile, a_reg)
        vf.store_align(dst_tile, reg, preg, a_reg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
