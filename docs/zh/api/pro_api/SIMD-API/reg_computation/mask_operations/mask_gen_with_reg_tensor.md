# vf.mask_gen_with_reg_tensor

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

从reg_tensor的指定数据块（DataBlock）的bit位生成mask_reg。

[reg_tensor](../basic_data_structures/reg_tensor.md)（256B）被划分为若干个DataBlock，offset参数指定从哪个DataBlock生成mask_reg。每个DataBlock中的每个bit会被broadcast到mask_reg中对应的多个bit位，broadcast倍数由数据类型位宽决定：

- **b16数据类型**（DT_FP16、DT_BF16、DT_INT16、DT_UINT16）：RegTensor划分为16个DataBlock（每个16B），每个bit broadcast到2bit，生成32B的mask_reg。offset取值范围为[0, 15]。
- **b32数据类型**（DT_FP32、DT_INT32、DT_UINT32）：RegTensor划分为32个DataBlock（每个8B），每个bit broadcast到4bit，生成32B的mask_reg。offset取值范围为[0, 31]。

### b16数据类型搬运原理

以b16数据类型为例，mask_gen_with_reg_tensor搬运原理如下图所示：

**图1** b16数据类型下mask_gen_with_reg_tensor搬运原理

![](../../../figures/mask_gen_with_reg_tensor_b16.jpg)

### b32数据类型搬运原理

以b32数据类型为例，mask_gen_with_reg_tensor搬运原理如下图所示：

**图2** b32数据类型下mask_gen_with_reg_tensor搬运原理

![](../../../figures/mask_gen_with_reg_tensor_b32.jpg)

## 函数原型

```python
mask_gen_with_reg_tensor(src, offset: Optional[int] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，reg_tensor。支持的数据类型请参见[约束说明](#约束说明)。 |
| offset | 输入 | 可选，指定从src的哪个DataBlock生成mask_reg，默认0。<br>- 16位宽数据类型时取值范围为[0, 15]（reg_tensor 256B划分为16个16B DataBlock）。<br>- 32位宽数据类型时取值范围为[0, 31]（reg_tensor 256B划分为32个8B DataBlock）。 |

## 约束说明

- 数据类型约束：

  源操作数支持的数据类型为：DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32。

## 返回值说明

返回dst目的操作数，[mask_reg](../basic_data_structures/mask_reg.md)。生成的mask_reg仅最低位有效：16位宽数据类型时每2bit中仅最低位有效，32位宽数据类型时每4bit中仅最低位有效。具体原理请参见[mask_reg工作原理](./create_mask.md#mask_reg工作原理)。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    reg = vf.load_align(src_tile, 0)
    dst = vf.mask_gen_with_reg_tensor(reg, offset=0)
    vf.store_align(dst_tile, reg, dst)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.full([1, 64], -1, device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
