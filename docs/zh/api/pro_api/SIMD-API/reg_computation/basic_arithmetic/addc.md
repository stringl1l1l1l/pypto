# vf.addc

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

该接口根据mask，对源操作数src0、src1及输入进位carry_src进行按元素求和操作，将结果写入目的操作数dst，同时将每个元素的进位结果写入carry。计算公式如下：

$$\{carry_i, dst_i\} = src0_i + src1_i + carry\_src_i$$

Carry flag（进位标志）用于表示加法进位，若src0、src1、carry_src输入按位相加后最高位有进位，在carry（存放进位的mask_reg）中对应位置每4bit设置1，否则写0。

以int64_t类型数据计算 -1 + 8 = 7为例，vf.addc接口的适用场景如下图所示：

**图1** vf.addc适用场景

![](../../../figures/addc_scenario.jpg)

## 函数原型

```python
addc(src0, src1, carry_src, preg) -> (carry, dst)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | 源操作数0，[reg_tensor](../reg_tensor.md)，源操作数src0、src1与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT32、DT_UINT32。 |
| src1 | 输入 | 源操作数1，[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。 |
| carry_src | 输入 | 输入进位值，[mask_reg](../mask_reg.md)。 |
| preg | 输入 | [mask_reg](../mask_reg.md)。 |

## 约束说明

无。

## 返回值说明

返回一个二元组 (carry, dst)。

- carry输出进位值，[mask_reg](../mask_reg.md)。
- dst目的操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    carry_src = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    carry = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_a = vf.load_align(src_a, 0, dtype=pl.DT_UINT32)
    reg_b = vf.load_align(src_b, 0, dtype=pl.DT_UINT32)
    carry, reg_out = vf.addc(reg_a, reg_b, carry_src, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 50, [1, 64], device=device, dtype=torch.int32)
    b = torch.randint(0, 50, [1, 64], device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    expected = (a.to(torch.int64) + b.to(torch.int64) + 1).to(torch.int32)
    assert torch.equal(out, expected)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
