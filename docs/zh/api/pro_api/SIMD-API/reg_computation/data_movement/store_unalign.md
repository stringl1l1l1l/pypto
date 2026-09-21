# vf.store_unalign

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

非对齐存储，将变长向量数据或mask_reg数据写入Tile。配合vf.unalign_reg_for_store()和vf.store_unalign_post()使用。当src为mask_reg时，后端自动分派mask_reg非对齐存储路径。

### 非对齐搬出原理

将源reg_tensor中的非对齐数据写入Tile地址dstAddr，根据ureg当前状态，分为两种场景：

**场景一：ureg为空**（第一次迭代）

如下图所示，从源reg_tensor（256B）读取数据，并将其搬运至目标Tile地址（dstAddr ~ 304）。处理流程如下：

① 调用**store_unalign**，此时ureg内无有效数据，表示连续非对齐搬出的起始状态，将源reg_tensor中对应Tile地址48 ~ 288的数据写入dstAddr。此外，源reg_tensor中对应Tile地址288 ~ 304的数据会被写入ureg。

② 调用**store_unalign_post**进行非对齐搬出后处理。将ureg中缓存的数据写入Tile地址288 ~ 304。

**图1**非对齐数据搬出（ureg为空）

![](../../../figures/unaligned_store_ureg_empty.jpg)

**场景二：ureg不为空**（除第一次迭代）

如下图所示，从源reg_tensor（256B）读取数据，并将其搬运至目标Tile地址（dstAddr ~ 304）。处理流程如下：

① 调用**store_unalign**，此时ureg内有有效数据，系统将ureg中Tile地址32 ~ dstAddr对应的数据与源reg_tensor中Tile地址dstAddr ~ 288对应的数据进行拼接，结果写入Tile地址dstAddr。此外，源reg_tensor中对应Tile地址288 ~ 304的数据会被写入ureg。

② 调用**store_unalign_post**进行非对齐搬出后处理。将ureg中缓存的数据写入Tile地址288 ~ 304。

**图2**非对齐数据搬出（ureg不为空）

![](../../../figures/unaligned_store_ureg_not_empty.jpg)

### 连续非对齐搬入搬出示例

**图3**连续非对齐搬入搬出示例（数据类型DT_UINT32）

![](../../../figures/contiguous_unaligned_load_store_unalign.jpg)

连续非对齐搬入时，vf.load_unalign会将后续未对齐的数据缓存至ureg，所以下一次搬入不需要再次调用vf.load_unalign_pre，只需在迭代开始前调用一次vf.load_unalign_pre，从而实现非对齐搬入的性能优化。

连续非对齐搬出时，下次迭代的vf.store_unalign会将本次迭代vf.store_unalign缓存至ureg中的数据写入Tile，所以本次迭代不需要调用vf.store_unalign_post将ureg数据写入Tile，只需在迭代结束后调用一次vf.store_unalign_post，从而实现非对齐搬出的性能优化。

## 函数原型

```python
store_unalign(tile, src, align_reg, stride=None, post_update: bool = False)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输出 | 目的操作数，Tile地址。 |
| src | 输入 | 源操作数，[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)类型，目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| align_reg | 输入 | alignment tracker寄存器（由vf.unalign_reg_for_store()创建）。 |
| stride | 输入 | 可选，存储元素个数或地址寄存器。<br>- 当为整型标量时，代表地址更新步长，仅post_update = True时有效。<br>- 当为AddrReg（由vf.create_addr_reg创建）时，使用向量偏移地址替代标量stride。src为reg_tensor时为必选输入；src为mask_reg时不传stride。 |
| post_update | 输入 | 可选，True时tracker自动累进到下一段，默认False。 |

## 约束说明

- 必须与vf.store_unalign_post()配对使用，在vf.store_unalign_post()之前调用。

## 返回值说明

无。

## 调用示例

### 基本非对齐存储

非对齐搬出与非对齐搬入配套使用，形成完整的非对齐数据搬运流程。vf.load_unalign_init分配非对齐搬入寄存器，vf.load_unalign_pre初始化缓存，vf.load_unalign执行搬入，vf.unalign_reg_for_store分配搬出对齐寄存器，vf.store_unalign执行搬出，vf.store_unalign_post刷出剩余数据。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    src_reg = vf.load_unalign(ureg, src_tile, post_update=True)
    store_ureg = vf.unalign_reg_for_store()
    vf.store_unalign(dst_tile, src_reg, store_ureg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, store_ureg, 0, post_update=True)

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
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### AddrReg非对齐存储示例

当stride参数传入AddrReg（由vf.create_addr_reg创建）时，AddrReg提供一组向量偏移地址，适用于变长步长的非对齐搬出场景。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    src_reg = vf.load_unalign(ureg, src_tile, post_update=True)
    store_ureg = vf.unalign_reg_for_store()
    for i in pl.range(0, 1, 1):
        addr_reg = vf.create_addr_reg(64, dtype=pl.DT_FP32)
        vf.store_unalign(dst_tile, src_reg, store_ureg, addr_reg, post_update=True)
        vf.store_unalign_post(dst_tile, store_ureg, addr_reg)

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
    assert out.shape == torch.Size([1, 64])

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### mask_reg非对齐存储示例

当src为mask_reg时，vf.store_unalign自动分派mask_reg非对齐存储路径。[mask_reg](../mask_reg.md) 32字节数据按16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）打包为16字节或按32位宽（DT_INT32、DT_UINT32、DT_FP32）打包为8字节写入Tile。硬件从每2bit（16位宽）/4bit（32位宽）中提取最低有效位(LSB)。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, mask_buf_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    cmp_mask = vf.ge(reg_a, 0.0, preg)
    ureg = vf.unalign_reg_for_store()
    vf.store_unalign(mask_buf_tile, cmp_mask, ureg)
    vf.store_unalign_post(mask_buf_tile, ureg, 0, post_update=True)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_mask_grp = pl.make_tile_group(type=tu, addrs=0x100, mutex_ids=[1])
    t_mask = t_mask_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(t_mask, out, [0, 0])
        example_vf(in_a, t_mask)
        pl.store(out, t_mask, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.ones([1, 64], device=device, dtype=torch.float32)
    out = torch.zeros([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert (out != 0).any(), "全正输入应产生非零打包掩码"
    a = -torch.ones([1, 64], device=device, dtype=torch.float32)
    out = torch.zeros([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert (out == 0).all(), "全负输入应产生零打包掩码"

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### INT64数据类型示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    ureg = vf.unalign_reg_for_store()
    vf.store_unalign(dst_tile, reg_a, ureg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, ureg, 0, post_update=True)
    reg_out = reg_a
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
