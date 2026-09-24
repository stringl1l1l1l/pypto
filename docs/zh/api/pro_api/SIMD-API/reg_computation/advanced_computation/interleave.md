# vf.interleave

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

给定源操作数src0和src1，将src0和src1中的元素交织存入结果操作数dst0和dst1中。交织排列方式如下图所示，其中每个方格代表一个元素：

$$dstReg0_{2i} = srcReg0_i, \quad dstReg1_{2i+1} = srcReg1_i$$

**图1** interleave交织示意图

![](../../../figures/interleave_data_layout.jpg)

## 函数原型

```python
interleave(src0, src1, dtype: Optional[DType] = None) -> (dst0, dst1)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或者[mask_reg](../basic_data_structures/mask_reg.md)类型。<br>- **reg_tensor输入**：源操作数src0、src1和目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。<br>- **mask_reg输入**：支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32。 |
| src1 | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或者[mask_reg](../basic_data_structures/mask_reg.md)类型，支持的数据类型和src0中的说明一致。 |
| dtype | 输入 | 可选，模板参数对应的数据类型。仅当源操作数为mask_reg时生效，决定按位交织的位宽。 |

## 约束说明

- src0和src1可以为同一个reg_tensor；dst0和dst1不能为同一个reg_tensor，配置为同一个reg_tensor会导致功能异常。

- 允许源操作数和目的操作数为同一个reg_tensor。

## 返回值说明

返回一个二元组 (dst0, dst1)。

- dst0目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或者[mask_reg](../basic_data_structures/mask_reg.md)类型，支持的数据类型和src0中的说明一致。
- dst1目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或者[mask_reg](../basic_data_structures/mask_reg.md)类型，支持的数据类型和src0中的说明一致。

## 调用示例

### reg_tensor调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    src0 = vf.load_align(src_a, 0)
    src1 = vf.load_align(src_b, 0)
    dst0, dst1 = vf.interleave(src0, src1)
    vf.store_align(dst_tile, dst0, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
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
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    b = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    assert out.shape == torch.Size([1, 128])

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### mask_reg调用示例

当源操作数为mask_reg时，vf.interleave按位交织两个mask_reg。交织位宽由mask_reg的数据类型决定。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    mask_full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    mask_m3 = vf.create_mask(pattern=pl.MaskPattern.M3, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    new_mask0, new_mask1 = vf.interleave(mask_full, mask_m3)
    new_mask0, new_mask1 = vf.de_interleave(new_mask0, new_mask1)
    reg_dst = vf.abs(reg, new_mask0)
    vf.store_align(dst_tile, reg_dst, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
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
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.abs(a), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### INT64数据类型示例

以下示例先对两个INT64寄存器执行`vf.interleave`，再对结果执行`vf.de_interleave`恢复原始数据，验证往返一致性。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    dst0, dst1 = vf.interleave(reg_a, reg_b)
    rec_a, rec_b = vf.de_interleave(dst0, dst1)
    vf.store_align(dst_tile, rec_a, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=512, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf_int64(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    b = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
