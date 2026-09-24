# vf.div

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

两个reg_tensor逐元素除法。

$$dstReg_i = srcReg0_i \div srcReg1_i$$

## 函数原型

```python
div(src0, src1, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | 源操作数0，[reg_tensor](../basic_data_structures/reg_tensor.md)，源操作数src0、src1与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32、DT_INT64、DT_UINT64。 |
| src1 | 输入 | 源操作数1，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型和src0中的说明一致。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)。 |
| mode | 输入 | 可选，对应[MergeMode](../basic_data_structures/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |
| precision | 输入 | 可选，高精度模式开关。True启用高精度模式；False（默认）为标准模式。高精度模式下支持的数据类型为：DT_FP16、DT_FP32。 |

## 约束说明

无。

## 返回值说明

返回dst目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型和src0中的说明一致。

## 调用示例

### 标准模式调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.div(reg_a, reg_b, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.abs(torch.randn([1, 64], device=device, dtype=torch.float32)) + 0.1
    b = torch.abs(torch.randn([1, 64], device=device, dtype=torch.float32)) + 0.1
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a / b, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### 高精度模式调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.div(reg_a, reg_b, preg, precision=True)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.abs(torch.randn([1, 64], device=device, dtype=torch.float32)) + 0.1
    b = torch.abs(torch.randn([1, 64], device=device, dtype=torch.float32)) + 0.1
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    ref = (a.cpu().double() / b.cpu().double()).float().to(device)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### FP16数据类型高精度示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp16(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.div(reg_a, reg_b, preg, precision=True)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_fp16(
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
        example_vf_fp16(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp16():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f'npu:{device_id}'
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.rand([1, 128], device=device, dtype=torch.float16) + 0.5
    b = torch.rand([1, 128], device=device, dtype=torch.float16) + 0.5
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_fp16[None, core_nums](a, b, out)
    torch.npu.synchronize()
    ref = (a.cpu().double() / b.cpu().double()).half().to(device)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

if __name__ == '__main__':
    test_example_fp16()
    print('PASSED')
```

### INT64数据类型示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile_a, src_tile_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile_a, 0)
    reg_b = vf.load_align(src_tile_b, 0)
    reg_out = vf.div(reg_a, reg_b, preg)
    vf.store_align(dst_tile, reg_out, preg)

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
    b = torch.randint(1, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.div(a, b, rounding_mode="trunc"), rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
