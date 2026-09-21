# vf.store_unalign_post

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

非对齐存储后处理，处理非对齐寄存器（UnalignRegForLoad）中剩余的未对齐字节。须在vf.store_unalign之后调用。

## 函数原型

```python
store_unalign_post(tile, align_reg, stride, post_update: bool = False)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输出 | 目的操作数，Tile地址。目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| align_reg | 输入 | alignment tracker寄存器（由vf.unalign_reg_for_store()创建）。 |
| stride | 输入 | 存储元素个数或地址寄存器。当为整型标量时，发射vstas指令（strided模式），post_update = True时同时作为地址更新步长，仅post_update = True时有效。当为AddrReg（由vf.create_addr_reg创建）时，发射vsta指令（AddrReg模式），须与vf.store_unalign的AddrReg模式（vstu）配对使用。 |
| post_update | 输入 | 可选，True时tracker自动累进，默认False。 |

## 约束说明

- 必须与vf.store_unalign配对使用，在vf.store_unalign之后调用。

- 如需基于vf.squeeze的有效元素个数进行非对齐存储后处理，请使用vf.squeeze_store_unalign_post接口。

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
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    reg = vf.load_unalign(ureg, src_tile, post_update=True)
    align_reg = vf.unalign_reg_for_store()
    vf.store_unalign(dst_tile, reg, align_reg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, align_reg, 0, post_update=True)

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
