# vf.mull

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

该接口根据mask对输入数据src0、src1按元素相乘操作，将乘法结果的低位部分写入dst_lo，溢出（高位）部分写入dst_hi。计算公式如下：

$$dst\_lo_i = (src0_i \times src1_i) \bmod 2^{bit}$$

$$dst\_hi_i = \lfloor (src0_i \times src1_i) / 2^{bit} \rfloor$$

其中，bit表示操作数的位宽bit数。

**图1** vf.mull功能说明

![](../../../figures/mull_diagram.jpg)

## 函数原型

```python
mull(src0, src1, preg) -> (dst_lo, dst_hi)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | 源操作数0，[reg_tensor](../reg_tensor.md)，源操作数src与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT32、DT_UINT32。 |
| src1 | 输入 | 源操作数1，[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。 |
| preg | 输入 | [mask_reg](../mask_reg.md)。 |

## 约束说明

无。

## 返回值说明

返回一个二元组 (dst_lo, dst_hi)。

- dst_lo 目的操作数（乘法结果低位），[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。
- dst_hi 目的操作数（乘法结果高位），[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_lo, dst_hi):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    reg_a = vf.load_align(src_a, 0, dtype=pl.DT_UINT32)
    reg_b = vf.load_align(src_b, 0, dtype=pl.DT_UINT32)
    reg_lo, reg_hi = vf.mull(reg_a, reg_b, preg)
    vf.store_align(dst_lo, reg_lo, preg)
    vf.store_align(dst_hi, reg_hi, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out_lo: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out_hi: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_lo_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_lo = t_lo_grp.current()
    t_hi_grp = pl.make_tile_group(type=tf, addrs=0x300, mutex_ids=[3])
    t_hi = t_hi_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf(in_a, in_b, t_lo, t_hi)
        pl.store(out_lo, t_lo, [0, 0])
        pl.store(out_hi, t_hi, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(1, 1000, [1, 64], device=device, dtype=torch.int32)
    b = torch.randint(1, 1000, [1, 64], device=device, dtype=torch.int32)
    out_lo = torch.empty([1, 64], device=device, dtype=torch.int32)
    out_hi = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, b, out_lo, out_hi)
    torch.npu.synchronize()
    product = a.to(torch.int64) * b.to(torch.int64)
    expected_lo = (product & 0xFFFFFFFF).to(torch.int32)
    expected_hi = (product >> 32).to(torch.int32)
    assert torch.equal(out_lo, expected_lo)
    assert torch.equal(out_hi, expected_hi)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
