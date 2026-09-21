# vf.pack

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

该指令会在后端根据src参数类型自动支持两种模式：

**reg_tensor输入**：将源操作数src中的元素选取低8位（对于16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）类型）、低16位（对于32位宽（DT_INT32、DT_UINT32、DT_FP32）类型）、低32位（对于64位宽（DT_INT64、DT_UINT64）类型），根据part选取的模式，写入dst的低半部分或高半部分。用于将宽类型数据压缩为窄类型数据。示意图如下图所示：

**图1** reg_tensor输入pack示意图

![reg_tensor输入pack示意图](../../../figures/pack_diagram.jpg)

**mask_reg输入**：将源操作数src中的偶数位，根据part选取的模式，提取到dst的低半部分或高半部分。示意图如下图所示：

**图2** mask_reg输入pack示意图

![mask_reg输入pack示意图](../../../figures/mask_pack_diagram.jpg)

## 函数原型

```python
pack(src, part: Optional[PackPart] = None, dtype: Optional[DType] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)类型。reg_tensor时数据类型为压缩前的宽类型，mask_reg时数据类型不变。 |
| part | 输入 | 可选，用于控制写入dst的低半部分还是高半部分，对应[PackPart](../types/PackPart.md)类型。<br>- pypto_pro.language.PackPart.LOWER：低位模式，写入dst的低半部分。<br>- pypto_pro.language.PackPart.UPPER：高位模式，写入dst的高半部分。<br>默认pypto_pro.language.PackPart.LOWER。双寄存器模式只支持pypto_pro.language.PackPart.LOWER模式。 |
| dtype | 输入 | 可选，数据类型。<br>- reg_tensor模式必选，指定目标reg_tensor的数据类型（如pypto_pro.language.DT_UINT8、pypto_pro.language.DT_UINT16等）。需要必选的原因在于将宽类型压缩为窄类型（如DT_UINT16→DT_UINT8），目标reg_tensor的数据类型与源reg_tensor不同，无法从源操作数推断，因此必须通过dtype参数显式指定目标数据类型。<br>- mask_reg模式保持寄存器类型，可省略。 |

## 约束说明

- 数据类型约束：

  - **reg_tensor输入**

    | dst | src |
    |---|---|
    | DT_UINT8 | DT_INT16 |
    | DT_UINT8 | DT_UINT16 |
    | DT_UINT16 | DT_INT32 |
    | DT_UINT16 | DT_UINT32 |
    | DT_UINT32 | DT_INT64 |
    | DT_UINT32 | DT_UINT64 |

  - **mask_reg输入**

    无约束

## 返回值说明

返回dst目的操作数，[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)类型。reg_tensor时数据类型为压缩后的窄类型，mask_reg时数据类型不变，支持的数据类型请参见[约束说明](#约束说明)。

## 调用示例

### reg_tensor调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    src = vf.load_align(src_tile, 0)
    dst = vf.pack(src, part=pl.PackPart.LOWER, dtype=pl.DT_UINT8)
    vf.store_align(dst_tile, dst, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
):
    tf_src = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    tf_dst = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_src, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_dst, addrs=0x100, mutex_ids=[1])
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
    a = torch.randint(0, 256, [1, 128], device=device, dtype=torch.int16)
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert out.dtype == torch.uint8

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### mask_reg调用示例

当源操作数为mask_reg时，vf.pack提取掩码的偶数位bit到低半部分或高半部分。mask_reg变体与reg_tensor变体共用part=参数指定模式。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    mask_a = vf.ge(reg, 0.0, preg)
    preg_packed = vf.pack(mask_a, part=pl.PackPart.LOWER)
    preg_unpacked = vf.unpack(preg_packed, part=pl.PackPart.LOWER)
    reg_dst = vf.abs(reg, preg_unpacked)
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
    expected = torch.where(a >= 0, a, torch.zeros_like(a))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### INT64数据类型示例

以下示例将INT64寄存器通过`vf.pack`压缩为UINT32（pack的目的数据类型须为无符号），再通过`vf.bit_cast`按位重解释为INT32，最后通过`vf.unpack`恢复为INT64，验证往返一致性。`vf.unpack`对INT32源执行符号扩展（高位填充符号位），正确还原原始INT64值。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    reg_packed = vf.pack(reg_a, dtype=pl.DT_UINT32)
    reg_signed = vf.bit_cast(reg_packed, dtype=pl.DT_INT32)
    reg_out = vf.unpack(reg_signed, dtype=pl.DT_INT64)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int64(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
