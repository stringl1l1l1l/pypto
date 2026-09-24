# vf.squeeze_store_unalign

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

非对齐存储，将变长向量数据写入Tile。基本功能和vf.store_unalign的reg_tensor模式一致，但是参数和约束有所区别。

- vf.store_unalign的reg_tensor模式下需要stride和post_update两个额外参数配置，而vf.squeeze_store_unalign没有这两个参数。

- 约束的区别请参考vf.store_unalign和vf.squeeze_store_unalign的约束说明章节。

## 函数原型

```python
squeeze_store_unalign(tile, src, align_reg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输出 | 目的操作数，Tile地址。 |
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)类型。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| align_reg | 输入 | alignment tracker寄存器（由vf.unalign_reg_for_store()创建）。 |

## 约束说明

- 调用前必须先调用vf.squeeze写入AR寄存器。

- 必须与vf.squeeze_store_unalign_post配对使用，在vf.squeeze_store_unalign_post之前调用。

- vf.squeeze和vf.squeeze_store_unalign必须交替使用，否则可能导致硬件挂死。

## 返回值说明

无。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    vf.clear_spr()
    reg_sq = vf.squeeze(reg_src, preg, gather_mode=pl.SqueezeMode.STORE_REG)
    align_reg = vf.unalign_reg_for_store()
    vf.squeeze_store_unalign(dst_tile, reg_sq, align_reg)
    vf.squeeze_store_unalign_post(dst_tile, align_reg)

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
    out = torch.zeros([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
