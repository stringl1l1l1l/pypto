# vf.get_mask_spr

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

从pypto_pro.language.set_vec_mask设置的掩码寄存器 {MASK1, MASK0} 中读取mask值，并按数据类型对应的格式转换后写入返回值[mask_reg](../mask_reg.md)。

具体转换方式：

- **32位宽（DT_INT32、DT_UINT32、DT_FP32）**：读取64bit的MASK0数据，将每个bit复制为4bit，写入mask_reg。
- **16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）**：读取完整128bit的 {MASK1, MASK0} 数据，将每个bit复制为2bit，写入mask_reg。

## 函数原型

```python
get_mask_spr(width: MaskWidth = MaskWidth.B32) -> preg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| width | 输入 | 可选，掩码宽度，决定读取的SPR位宽及扩展方式，对应[MaskWidth](../types/MaskWidth.md)类型。<br>- pypto_pro.language.MaskWidth.B32（默认）：读取64bit MASK0，每bit扩展为4bit。<br>- pypto_pro.language.MaskWidth.B16：读取128bit {MASK1, MASK0}，每bit扩展为2bit。 |

## 约束说明

- 数据类型约束：

  | width | 返回值 |
  |---|---|
  | pypto_pro.language.MaskWidth.B32 | mask_reg（对应32位宽：DT_INT32、DT_UINT32、DT_FP32） |
  | pypto_pro.language.MaskWidth.B16 | mask_reg（对应16位宽：DT_INT16、DT_UINT16、DT_FP16、DT_BF16） |

- 本接口为兼容性接口，建议优先采用vf.create_mask和vf.update_mask进行mask_reg计算。

- 本接口使用前需选择与掩码含义一致的模式，并通过pypto_pro.language.set_vec_mask或产生掩码的VF指令设置SPR {MASK1, MASK0}。按位掩码使用norm模式；元素计数使用count模式。

## 返回值说明

返回mask_reg，从SPR {MASK1, MASK0} 读取并转换。

## 调用示例

先用pypto_pro.language.set_vec_mask设置SPR {MASK1, MASK0}，再用vf.get_mask_spr读取到mask_reg，最后用该mask_reg控制计算：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    spr_mask = vf.get_mask_spr(width=pl.MaskWidth.B32)
    reg_dst = vf.abs(reg, spr_mask)
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
        pl.set_mask_norm()
        pl.set_vec_mask(0, 0xFFFFFFFF)
        example_vf(in_a, t_out)
        pl.reset_mask()
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
    expected = torch.zeros_like(a)
    expected[:, :32] = torch.abs(a[:, :32])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
