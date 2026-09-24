# vf.sqrt

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

该接口根据mask，对源操作数src逐元素做平方根运算，将结果写入目的操作数dst。计算公式如下：

$$dst_i = (src_i)^{1/2}$$

## 函数原型

```python
sqrt(src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，必须为非负数，源操作数src与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_FP16、DT_FP32。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)。 |
| mode | 输入 | 可选，对应[MergeMode](../basic_data_structures/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |
| precision | 输入 | 可选，高精度模式开关。True启用高精度模式；False（默认）为标准模式。 |

## 约束说明

无。

## 返回值说明

返回dst目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型和src中的说明一致。

## 调用示例

### 标准模式调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.sqrt(reg_a, preg)
    vf.store_align(dst_tile, reg_out, preg)

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
    a = torch.abs(torch.randn([1, 64], device=device, dtype=torch.float32)) + 0.1
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.sqrt(a), rtol=1e-5, atol=1e-5)

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
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.sqrt(reg_a, preg, precision=True)
    vf.store_align(dst_tile, reg_out, preg)

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
    a = torch.linspace(1e-40, 1e-39, 64, device=device, dtype=torch.float32).reshape(1, 64)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    ref = torch.sqrt(a.cpu().double()).float().to(device)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
